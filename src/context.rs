// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

use crate::locked::*;
use cranelift_jit::JITModule;
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem;
use std::rc::Rc;
use std::sync::Arc;
use synfx_dsp::AtomicFloat;

/// Default size of undeclared buffers.
pub const BUFFER_DEFAULT_SIZE: usize = 16;

/// Auxilary variables to access directly from the machine code.
pub(crate) const AUX_VAR_COUNT: usize = 3;

pub(crate) const AUX_VAR_IDX_SRATE: usize = 0;
pub(crate) const AUX_VAR_IDX_ISRATE: usize = 1;
pub(crate) const AUX_VAR_IDX_RESET: usize = 2;

pub enum DSPNodeContextError {
    UnknownTable(usize),
    WrongTableSize { tbl_idx: usize, new_size: usize, old_size: usize },
}

/// Configures the environment that will be available to the [DSPFunction]
/// that is provided by [DSPNodeContext].
///
/// This could for instance be the number of atoms to be used by `atomr`/`atomw`, the
/// number and length of buffers or the audio samples...
#[derive(Debug, Clone)]
pub struct DSPContextConfig {
    /// The number of atoms available to `atomr`/`atomw`.
    pub atom_count: usize,
    /// The number of buffers available to `bufr`/`bufw`.
    pub buffer_count: usize,
    /// The number of available tables for the `tblr`/`tblw` operations.
    /// The tables can be swapped out at runtime using the [DSPNodeContext::send_table] method.
    pub tables: Vec<Arc<Vec<f32>>>,
}

impl Default for DSPContextConfig {
    fn default() -> Self {
        Self {
            atom_count: 512,
            buffer_count: 16,
            tables: vec![Arc::new(vec![0.0; 16])],
        }
    }
}

/// This table holds all the DSP state including the state of the individual DSP nodes
/// that were created by the [crate::jit::DSPFunctionTranslator].
pub struct DSPNodeContext {
    /// The environment configuration for the [DSPFunction] to operate in.
    pub(crate) config: DSPContextConfig,
    /// The global DSP state that is passed to all stateful DSP nodes.
    state: *mut DSPState,
    /// Persistent variables:
    persistent_var_index: usize,
    /// An assignment of persistent variables to their index in the `persistent_vars` vector.
    persistent_var_map: HashMap<String, usize>,
    /// A map of unique DSP node instances (identified by dsp_node_uid) that need private state.
    node_states: HashMap<u64, Box<DSPNodeState>>,
    /// A generation counter to determine whether some [DSPNodeState] instances in `node_states`
    /// can be cleaned up.
    generation: u64,
    /// Contains the currently compiled [DSPFunction].
    next_dsp_fun: Option<Box<DSPFunction>>,
    /// If enabled, some extra data will be collected.
    debug_enabled: bool,
    /// If [DSPNodeContext::set_debug] is enabled, this contains the most recently compiled piece
    /// of cranelift intermedite representation. You can receive this via [DSPNodeContext::get_ir_dump].
    pub(crate) cranelift_ir_dump: String,
    /// An array of atomic floats to exchange data with the live real time thread.
    /// These AtomicFloats will be shared via the [DSPState] structure and read/written using
    /// the `atomw` and `atomr` nodes.
    atoms: Vec<Arc<AtomicFloat>>,
    /// Holds the current buffer lengths, they are updated
    /// in [DSPNodeContext::finalize_dsp_function].
    buffer_lengths: Vec<usize>,
    /// Holds the most recently declared buffer lengths, these are used to determine
    /// if we need to send a buffer update with the [DSPFunction]
    /// in [DSPNodeContext::finalize_dsp_function].
    pub(crate) buffer_declare: Vec<usize>,
}

impl DSPNodeContext {
    fn new() -> Self {
        Self::new_with_config(DSPContextConfig::default())
    }

    fn new_with_config(config: DSPContextConfig) -> Self {
        let mut atoms = vec![];
        atoms.resize_with(config.atom_count, || Arc::new(AtomicFloat::new(0.0)));
        let atoms_state = atoms.clone();

        let mut buffer_lengths = vec![];
        let mut buffers = vec![];
        for _ in 0..config.buffer_count {
            buffers.push(vec![0.0; BUFFER_DEFAULT_SIZE]);
            buffer_lengths.push(BUFFER_DEFAULT_SIZE);
        }
        let buffers = LockedMutPtrs::new(buffers);
        let buffer_declare = buffer_lengths.clone();

        let tables = LockedPtrs::new(config.tables.clone());

        Self {
            config,
            state: Box::into_raw(Box::new(DSPState {
                x: 0.0,
                y: 0.0,
                srate: 44100.0,
                israte: 1.0 / 44100.0,
                atoms: atoms_state,
                buffers,
                tables,
            })),
            node_states: HashMap::new(),
            generation: 0,
            next_dsp_fun: None,
            persistent_var_map: HashMap::new(),
            persistent_var_index: 0,
            debug_enabled: false,
            cranelift_ir_dump: String::from(""),
            atoms,
            buffer_lengths,
            buffer_declare,
        }
    }

    /// Creates a new [DSPNodeContext] that you can pass into [crate::JIT::new].
    pub fn new_ref() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self::new()))
    }

    pub(crate) fn init_dsp_function(&mut self) {
        self.generation += 1;
        self.next_dsp_fun = Some(Box::new(DSPFunction::new(self.state, self.generation)));
    }

    /// Enabled debug information collection. See also [DSPNodeContext::get_ir_dump].
    pub fn set_debug(&mut self, enabled: bool) {
        self.debug_enabled = enabled;
    }

    /// Returns if debug is enabled.
    pub fn debug_enabled(&self) -> bool {
        self.debug_enabled
    }

    /// If [DSPNodeContext::set_debug] is enabled, this will return the most recent
    /// IR code for the most recently compiled [DSPFunction].
    pub fn get_ir_dump(&self) -> &str {
        &self.cranelift_ir_dump
    }

    /// Returns you a reference to the specified atom connected with the DSP backend.
    /// These atoms can be read and written in the [DSPFunction] using the `atomr` and `atomw`
    /// nodes.
    pub fn atom(&self, idx: usize) -> Option<Arc<AtomicFloat>> {
        self.atoms.get(idx).cloned()
    }

    /// Retrieve the index into the most recently compiled [DSPFunction].
    /// To be used by [DSPFunction::access_persistent_var].
    pub fn get_persistent_variable_index_by_name(&self, pers_var_name: &str) -> Option<usize> {
        self.persistent_var_map.get(pers_var_name).map(|i| *i)
    }

    /// Retrieve the index into the persistent variable vector passed in as "&pv".
    pub(crate) fn get_persistent_variable_index(
        &mut self,
        pers_var_name: &str,
    ) -> Result<usize, String> {
        let index = if let Some(index) = self.persistent_var_map.get(pers_var_name) {
            *index
        } else {
            let index = self.persistent_var_index;
            self.persistent_var_index += 1;
            self.persistent_var_map.insert(pers_var_name.to_string(), index);
            index
        };

        if let Some(next_dsp_fun) = &mut self.next_dsp_fun {
            next_dsp_fun.touch_persistent_var_index(index);
            Ok(index)
        } else {
            Err("No DSPFunction in DSPNodeContext".to_string())
        }
    }

    /// Tries to send a new table to the backend. You have to make sure the table
    /// has exactly the same size as the previous table given in the [DSPContextConfig].
    /// Otherwise a [DSPNodeContextError] is returned.
    pub fn send_table(
        &mut self,
        tbl_idx: usize,
        table: Arc<Vec<f64>>,
    ) -> Result<(), DSPNodeContextError> {
        let config_tbl_len = 0;

        // Err(DSPNodeContextError::UnknwonTable(tbl_idx)

        Err(DSPNodeContextError::WrongTableSize {
            tbl_idx,
            new_size: table.len(),
            old_size: config_tbl_len,
        })
    }

    /// Adds a [DSPNodeState] to the currently compiled [DSPFunction] and returns
    /// the index into the node state vector in the [DSPFunction], so that the JIT
    /// code can index into that vector to find the right state pointer.
    pub(crate) fn add_dsp_node_instance(
        &mut self,
        node_type: Arc<dyn DSPNodeType>,
        dsp_node_uid: u64,
    ) -> Result<usize, String> {
        if let Some(next_dsp_fun) = &mut self.next_dsp_fun {
            if next_dsp_fun.has_dsp_node_state_uid(dsp_node_uid) {
                return Err(format!(
                    "node_state_uid has been used multiple times in same AST: {}",
                    dsp_node_uid
                ));
            }

            if !self.node_states.contains_key(&dsp_node_uid) {
                self.node_states.insert(
                    dsp_node_uid,
                    Box::new(DSPNodeState::new(dsp_node_uid, node_type.clone())),
                );
            }

            if let Some(state) = self.node_states.get_mut(&dsp_node_uid) {
                if state.node_type().name() != node_type.name() {
                    return Err(format!(
                        "Different DSPNodeType for uid {}: {} != {}",
                        dsp_node_uid,
                        state.node_type().name(),
                        node_type.name()
                    ));
                }

                Ok(next_dsp_fun.install(state))
            } else {
                Err(format!("NodeState does not exist, but it should... bad! {}", dsp_node_uid))
            }
        } else {
            Err("No DSPFunction in DSPNodeContext".to_string())
        }
    }

    pub(crate) fn finalize_dsp_function(
        &mut self,
        function_ptr: *const u8,
        module: JITModule,
    ) -> Option<Box<DSPFunction>> {
        if let Some(mut next_dsp_fun) = self.next_dsp_fun.take() {
            for (i, (len, declare)) in self.buffer_lengths.iter().zip(self.buffer_declare.iter()).enumerate() {
                if *len != *declare {
                    next_dsp_fun.add_buffer_update(i, *declare);
                }
            }

            for (len, declare) in self.buffer_lengths.iter_mut().zip(self.buffer_declare.iter_mut()) {
                *len = *declare;
            }

            next_dsp_fun.set_function_ptr(function_ptr, module);

            for (_, node_state) in self.node_states.iter_mut() {
                node_state.set_initialized();
            }

            Some(next_dsp_fun)
        } else {
            None
        }
    }

    /// If you received a [DSPFunction] back from the audio thread, you should
    /// pass it into this function. It will make sure to purge old unused [DSPNodeState] instances.
    pub fn cleanup_dsp_fun_after_user(&mut self, _fun: Box<DSPFunction>) {
        // TODO: Garbage collect and free unused node state!
        //       But this must happen by the backend/frontend thread separation.
        //       Best would be to provide DSPNodeContext::cleaup_dsp_function_after_use(DSPFunction).
    }

    /// You must call this after all [DSPFunction] instances compiled with this state are done executing.
    /// If you don't call this, you might get a memory leak.
    /// The API is a bit manual at this point, because usually [DSPFunction]
    /// will be executed on a different thread, and synchronizing this would come with
    /// additional overhead that I wanted to save.
    pub fn free(&mut self) {
        if !self.state.is_null() {
            unsafe { Box::from_raw(self.state) };
            self.state = std::ptr::null_mut();
        }
    }
}

impl Drop for DSPNodeContext {
    fn drop(&mut self) {
        if !self.state.is_null() {
            eprintln!("WBlockDSP JIT DSPNodeContext not cleaned up on exit. Forgot to call free() or keep it alive long enough?");
        }
    }
}

/// This structure holds all the [DSPNodeType] definitions and provides
/// them to the [crate::JIT] and [crate::jit::DSPFunctionTranslator].
pub struct DSPNodeTypeLibrary {
    type_by_name: HashMap<String, Arc<dyn DSPNodeType>>,
    types: Vec<Arc<dyn DSPNodeType>>,
}

impl DSPNodeTypeLibrary {
    /// Create a new instance of this.
    pub fn new() -> Self {
        Self { types: vec![], type_by_name: HashMap::new() }
    }

    /// Add the given [DSPNodeType] to this library.
    pub fn add(&mut self, typ: Arc<dyn DSPNodeType>) {
        self.types.push(typ.clone());
        self.type_by_name.insert(typ.name().to_string(), typ);
    }

    /// Retrieves a [DSPNodeType] by it's name.
    pub fn get_type_by_name(&self, typ_name: &str) -> Option<Arc<dyn DSPNodeType>> {
        self.type_by_name.get(typ_name).cloned()
    }

    /// Iterate through all types in the Library:
    ///
    ///```
    /// use synfx_dsp_jit::*;
    ///
    /// let lib = DSPNodeTypeLibrary::new();
    /// // ...
    /// lib.for_each(|typ| -> Result<(), ()> {
    ///     println!("Type available: {}", typ.name());
    ///     Ok(())
    /// }).expect("no error");
    ///```
    pub fn for_each<T, F: FnMut(&Arc<dyn DSPNodeType>) -> Result<(), T>>(
        &self,
        mut f: F,
    ) -> Result<(), T> {
        for t in self.types.iter() {
            f(t)?;
        }
        Ok(())
    }
}

/// This macro can help you defining new stateful DSP nodes.
///
///```
/// use synfx_dsp_jit::*;
///
/// struct MyDSPNode {
///     value: f64,
/// }
///
/// impl MyDSPNode {
///     fn reset(&mut self, _state: &mut DSPState) {
///         *self = Self::default();
///     }
/// }
///
/// impl Default for MyDSPNode {
///     fn default() -> Self {
///         Self { value: 0.0 }
///     }
/// }
///
/// extern "C" fn process_my_dsp_node(my_state: *mut MyDSPNode) -> f64 {
///     let mut my_state = unsafe { &mut *my_state };
///     my_state.value += 1.0;
///     my_state.value
/// }
///
/// // DIYNodeType is the type that is newly defined here, that one you
/// // pass to DSPNodeTypeLibrary::add
/// synfx_dsp_jit::stateful_dsp_node_type! {
///     DIYNodeType, MyDSPNode => process_my_dsp_node "my_dsp" "Sr"
///     doc
///     "This is a simple counter. It counts by increments of 1.0 everytime it's called."
///     inputs
///     outputs
///     0 "sum"
/// }
///
/// // Then use the type by adding it:
/// fn make_library() -> DSPNodeTypeLibrary {
///     let mut lib = DSPNodeTypeLibrary::new();
///     lib.add(DIYNodeType::new_ref());
///     lib
/// }
///```
///
/// You might've guessed, `process_my_dsp_node` is the function identifier in the Rust
/// code. The `"my_dsp"` is the name you can use to refer to this in [crate::ASTNode::Call]:
/// `ASTNode::Call("my_dsp".to_string(), 1, ...)`.
/// **Attention:** Make sure to provide unique state IDs here!
///
/// The `"Sr"` is a string that specifies the signature of the function. Following characters
/// are available:
///
/// - "v" - A floating point value
/// - "D" - The global [crate::DSPState] pointer
/// - "S" - The node specific state pointer (`MyDSPNode`)
/// - "M" - A pointer to the multi return value array, of type `*mut [f64; 5]`. These can be accessed
/// by the variables "%1" to "%5" after the call.
/// - "r" - Must be specified as last one, defines that this function returns something.
///
#[macro_export]
macro_rules! stateful_dsp_node_type {
    ($node_type: ident, $struct_type: ident =>
     $func_name: ident $jit_name: literal $signature: literal
     doc $doc: literal
     inputs $($idx: literal $inp: literal)*
     outputs $($idxo: literal $out: literal)*) => {
        struct $node_type;
        impl $node_type {
            fn new_ref() -> std::sync::Arc<Self> {
                std::sync::Arc::new(Self {})
            }
        }
        impl DSPNodeType for $node_type {
            fn name(&self) -> &str {
                $jit_name
            }

            fn function_ptr(&self) -> *const u8 {
                $func_name as *const u8
            }

            fn signature(&self, i: usize) -> Option<DSPNodeSigBit> {
                match $signature.chars().nth(i) {
                    Some('v') => Some(DSPNodeSigBit::Value),
                    Some('D') => Some(DSPNodeSigBit::DSPStatePtr),
                    Some('S') => Some(DSPNodeSigBit::NodeStatePtr),
                    Some('M') => Some(DSPNodeSigBit::MultReturnPtr),
                    _ => None,
                }
            }

            fn has_return_value(&self) -> bool {
                $signature.find("r").is_some()
            }

            fn reset_state(&self, dsp_state: *mut DSPState, state_ptr: *mut u8) {
                let ptr = state_ptr as *mut $struct_type;
                unsafe {
                    (*ptr).reset(&mut (*dsp_state));
                }
            }

            fn allocate_state(&self) -> Option<*mut u8> {
                Some(Box::into_raw(Box::new($struct_type::default())) as *mut u8)
            }

            fn deallocate_state(&self, ptr: *mut u8) {
                unsafe { Box::from_raw(ptr as *mut $struct_type) };
            }

            fn documentation(&self) -> &str {
                $doc
            }

            fn input_names(&self, index: usize) -> Option<&str> {
                match index {
                    $($idx => Some($inp),)*
                    _ => None
                }
            }

            fn input_index_by_name(&self, name: &str) -> Option<usize> {
                match name {
                    $($inp => Some($idx),)*
                    _ => None
                }
            }

            fn output_names(&self, index: usize) -> Option<&str> {
                match index {
                    $($idxo => Some($out),)*
                    _ => None
                }
            }

            fn output_index_by_name(&self, name: &str) -> Option<usize> {
                match name {
                    $($out => Some($idxo),)*
                    _ => None
                }
            }
        }
    };
}

/// This macro can help you defining new stateless DSP nodes.
///
///```
/// use synfx_dsp_jit::*;
///
/// extern "C" fn process_mul2(v: f64) -> f64 {
///     v * 2.0
/// }
///
/// synfx_dsp_jit::stateless_dsp_node_type! {
///     Mul2NodeType => process_mul2 "mul2" "vr"
///     doc
///     "A simple multiplication by 2.0. Using '*' is simpler thought..."
///     inputs
///     0 ""
///     outputs
///     0 ""
/// }
///
/// // Then use the type by adding it:
/// fn make_library() -> DSPNodeTypeLibrary {
///     let mut lib = DSPNodeTypeLibrary::new();
///     lib.add(Mul2NodeType::new_ref());
///     lib
/// }
///```
///
/// The `"vr"` is a string that specifies the signature of the function. Following characters
/// are available:
///
/// - "v" - A floating point value
/// - "D" - The global [crate::DSPState] pointer
/// - "M" - A pointer to the multi return value array, of type `*mut [f64; 5]`. These can be accessed
/// by the variables "%1" to "%5" after the call.
/// - "r" - Must be specified as last one, defines that this function returns something.
///
#[macro_export]
macro_rules! stateless_dsp_node_type {
    ($node_type: ident =>
     $func_name: ident $jit_name: literal $signature: literal
     doc $doc: literal
     inputs $($idx: literal $inp: literal)*
     outputs $($idxo: literal $out: literal)*) => {
        #[derive(Default)]
        struct $node_type;
        impl $node_type {
            #[allow(dead_code)]
            fn new_ref() -> std::sync::Arc<Self> {
                std::sync::Arc::new(Self {})
            }
        }
        impl DSPNodeType for $node_type {
            fn name(&self) -> &str {
                $jit_name
            }

            fn function_ptr(&self) -> *const u8 {
                $func_name as *const u8
            }

            fn signature(&self, i: usize) -> Option<DSPNodeSigBit> {
                match $signature.chars().nth(i) {
                    Some('v') => Some(DSPNodeSigBit::Value),
                    Some('D') => Some(DSPNodeSigBit::DSPStatePtr),
                    Some('M') => Some(DSPNodeSigBit::MultReturnPtr),
                    _ => None,
                }
            }

            fn has_return_value(&self) -> bool {
                $signature.find("r").is_some()
            }

            fn documentation(&self) -> &str {
                $doc
            }

            fn input_names(&self, index: usize) -> Option<&str> {
                match index {
                    $($idx => Some($inp),)*
                    _ => None
                }
            }

            fn input_index_by_name(&self, name: &str) -> Option<usize> {
                match name {
                    $($inp => Some($idx),)*
                    _ => None
                }
            }

            fn output_names(&self, index: usize) -> Option<&str> {
                match index {
                    $($idxo => Some($out),)*
                    _ => None
                }
            }

            fn output_index_by_name(&self, name: &str) -> Option<usize> {
                match name {
                    $($out => Some($idxo),)*
                    _ => None
                }
            }
        }
    };
}

/// This is the result of the JIT compiled [crate::ast::ASTNode] tree.
/// You can send this structure to the audio backend thread and execute it
/// using [DSPFunction::exec].
///
/// To execute this [DSPFunction] properly, you have to call [DSPFunction::init]
/// once the newly allocated structure is received by the DSP executing thread.
///
/// If the sample rate changes or the stateful DSP stuff must be resetted,
/// you should call [DSPFunction::reset] or [DSPFunction::set_sample_rate].
/// Of course also only on the DSP executing thread.
pub struct DSPFunction {
    state: *mut DSPState,
    /// Contains the types of the corresponding `node_states`. The [DSPNodeType] is
    /// necessary to reset the state pointed to by the pointers in `node_states`.
    node_state_types: Vec<Arc<dyn DSPNodeType>>,
    /// Contains the actual pointers to the state that was constructed by the corresponding [DSPNodeState].
    node_states: Vec<*mut u8>,
    /// Constains indices into `node_states`, so that they can be reset/initialized by [DSPFunction::init].
    /// Only contains recently added (as determined by [DSPNodeContext]) and uninitialized state indices.
    node_state_init_reset: Vec<usize>,
    /// Keeps the node_state_uid of the [DSPNodeState] pieces used already in this
    /// function. It's for error detection when building this [DSPFunction], to prevent
    /// the user from evaluating a stateful DSP node multiple times.
    node_state_uids: Vec<u64>,
    /// Generation of the corresponding [DSPNodeContext].
    dsp_ctx_generation: u64,
    /// The JITModule that is the home for the `function` pointer. It must be kept alive
    /// as long as the `function` pointer is in use.
    module: Option<JITModule>,
    /// Storage of persistent variables:
    persistent_vars: Vec<f64>,
    /// Buffer updates for the buffers in [DSPState], these are determined and set
    /// in [DSPNodeContext::finalize_dsp_function].
    buffer_updates: Vec<(usize, Vec<f64>)>,
    /// This is just a flag as precaution, in case init() is accidentally called
    /// multiple times.
    buffer_updates_done: bool,
    /// Auxilary variables to access directly from the machine code. Holds information such as
    /// the sample rate or the inverse of the sample rate.
    aux_vars: [f64; AUX_VAR_COUNT],
    /// Is true directly after reset.
    resetted: bool,
    function: Option<
        fn(
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            *mut f64,
            *mut f64,
            *mut f64,
            *mut DSPState,
            *const *mut u8,
            *mut f64,
            *mut f64,
            *const *mut f64,
            *const u64,
            *const *const f32,
            *const u64,
        ) -> f64,
    >,
}

unsafe impl Send for DSPFunction {}
unsafe impl Sync for DSPFunction {}

impl DSPFunction {
    /// Used by [DSPNodeContext] to create a new instance of this.
    pub(crate) fn new(state: *mut DSPState, dsp_ctx_generation: u64) -> Self {
        Self {
            state,
            node_state_types: vec![],
            node_states: vec![],
            node_state_init_reset: vec![],
            node_state_uids: vec![],
            persistent_vars: vec![],
            aux_vars: [0.0; AUX_VAR_COUNT],
            function: None,
            dsp_ctx_generation,
            module: None,
            resetted: false,
            buffer_updates: vec![],
            buffer_updates_done: true,
        }
    }

    /// At the end of the compilation the [crate::JIT] will put the resulting function
    /// pointer into this function.
    pub(crate) fn set_function_ptr(&mut self, function: *const u8, module: JITModule) {
        self.module = Some(module);
        self.function = Some(unsafe {
            mem::transmute::<
                _,
                fn(
                    f64,
                    f64,
                    f64,
                    f64,
                    f64,
                    f64,
                    *mut f64,
                    *mut f64,
                    *mut f64,
                    *mut DSPState,
                    *const *mut u8,
                    *mut f64,
                    *mut f64,
                    *const *mut f64,
                    *const u64,
                    *const *const f32,
                    *const u64,
                ) -> f64,
            >(function)
        });
    }

    /// Appends a buffer update to this [DSPFunction], to update the buffers
    /// according to [crate::ast::ASTNode::BufDeclare]. Buffers are only updated
    /// if they get a new length though.
    pub(crate) fn add_buffer_update(&mut self, buf_idx: usize, length: usize) {
        self.buffer_updates.push((buf_idx, vec![0.0; length]));
        self.buffer_updates_done = false;
    }

    /// This function must be called before [DSPFunction::exec]!
    /// otherwise your states might not be properly initialized or preserved.
    ///
    /// If you recompiled a function, pass the old one on the audio thread to
    /// the `previous_function` parameter here. It will take care of preserving
    /// state, such as persistent variables (those that start with "*": `crate::build::var("*abc")`).
    pub fn init(&mut self, srate: f64, previous_function: Option<&DSPFunction>) {
        if let Some(previous_function) = previous_function {
            let prev_len = previous_function.persistent_vars.len();
            let now_len = self.persistent_vars.len();
            let len = prev_len.min(now_len);
            self.persistent_vars[0..len].copy_from_slice(&previous_function.persistent_vars[0..len])
        } else {
            self.resetted = true;
        }

        if !self.buffer_updates_done {
            for (idx, new_vec) in self.buffer_updates.iter_mut() {
                unsafe {
                    let old_len = (*self.state).buffers.element_len(*idx);
                    let old_vec = (*self.state).buffers.pointers()[*idx];
                    let min_len = old_len.min(new_vec.len());
                    std::ptr::copy_nonoverlapping(old_vec, new_vec.as_mut_ptr(), min_len);
                    let _ = (*self.state).buffers.swap_element(*idx, new_vec);
                }
            }
            self.buffer_updates_done = true;
        }

        unsafe {
            (*self.state).srate = srate;
            (*self.state).israte = 1.0 / srate;
        }
        self.aux_vars[AUX_VAR_IDX_SRATE] = srate;
        self.aux_vars[AUX_VAR_IDX_ISRATE] = 1.0 / srate;

        for idx in self.node_state_init_reset.iter() {
            let typ = &self.node_state_types[*idx as usize];
            let ptr = self.node_states[*idx as usize];
            typ.reset_state(self.state, ptr);
        }
    }

    /// If the audio thread changes the sampling rate, call this function, it will update
    /// the [DSPState] and reset all [DSPNodeState]s.
    pub fn set_sample_rate(&mut self, srate: f64) {
        unsafe {
            (*self.state).srate = srate;
            (*self.state).israte = 1.0 / srate;
        }
        self.aux_vars[AUX_VAR_IDX_SRATE] = srate;
        self.aux_vars[AUX_VAR_IDX_ISRATE] = 1.0 / srate;

        self.reset();
    }

    /// If the DSP state needs to be resetted, call this on the audio thread.
    pub fn reset(&mut self) {
        self.resetted = true;
        for (typ, ptr) in self.node_state_types.iter().zip(self.node_states.iter_mut()) {
            typ.reset_state(self.state, *ptr);
        }
        self.persistent_vars.fill(0.0);
    }

    /// Use this to retrieve a pointer to the [DSPState] to access it between
    /// calls to [DSPFunction::exec].
    pub fn get_dsp_state_ptr(&self) -> *mut DSPState {
        self.state
    }

    /// Use this to access the [DSPState] pointer between calls to [DSPFunction::exec].
    ///
    /// # Safety
    ///
    /// You must not create multiple aliasing references from that DSP state!
    pub unsafe fn with_dsp_state<R, F: FnMut(*mut DSPState) -> R>(&mut self, mut f: F) -> R {
        f(self.get_dsp_state_ptr())
    }

    /// Use this to access the state of a specific DSP node state pointer between
    /// calls to [DSPFunction::exec].
    ///
    /// The `node_state_uid` and the type you pass here must match! It's your responsibility
    /// to make sure this works!
    ///
    /// # Safety
    ///
    /// You absolutely must know which ID has which [DSPNodeType], otherwise this will badly go wrong!
    ///
    ///```
    /// use synfx_dsp_jit::*;
    /// use synfx_dsp_jit::build::*;
    /// use synfx_dsp_jit::stdlib::AccumNodeState;
    ///
    /// let (ctx, mut fun) = instant_compile_ast(call("accum", 21, &[var("in1"), literal(0.0)])).unwrap();
    ///
    /// fun.init(44100.0, None);
    /// // Accumulate 42.0 here:
    /// fun.exec_2in_2out(21.0, 0.0);
    /// fun.exec_2in_2out(21.0, 0.0);
    ///
    /// unsafe {
    ///     // Check 42.0 and set 99.0
    ///     fun.with_node_state(21, |state: *mut AccumNodeState| {
    ///         assert!(((*state).value - 42.0).abs() < 0.0001);
    ///         (*state).value = 99.0;
    ///     })
    /// };
    ///
    /// // Accumulate up to 100.0 here:
    /// let (_, _, ret) = fun.exec_2in_2out(1.0, 0.0);
    /// assert!((ret - 100.0).abs() < 0.0001);
    ///
    /// ctx.borrow_mut().free();
    ///```
    #[allow(clippy::result_unit_err)]
    pub unsafe fn with_node_state<T, R, F: FnMut(*mut T) -> R>(
        &mut self,
        node_state_uid: u64,
        mut f: F,
    ) -> Result<R, ()> {
        if let Some(state_ptr) = self.get_node_state_ptr(node_state_uid) {
            Ok(f(state_ptr as *mut T))
        } else {
            Err(())
        }
    }

    /// Retrieves the DSP node state pointer for a certain unique node state id.
    ///
    /// # Safety
    ///
    /// You are responsible afterwards for knowing what type the actual pointer is of.
    pub fn get_node_state_ptr(&self, node_state_uid: u64) -> Option<*mut u8> {
        for (i, uid) in self.node_state_uids.iter().enumerate() {
            if *uid == node_state_uid {
                return Some(self.node_states[i]);
            }
        }

        None
    }

    /// Helper function, it lets you specify only the contents of the parameters
    /// `"in1"` and `"in2"`. It also returns you the values for `"&sig1"` and `"&sig2"`
    /// after execution. The third value is the return value of the compiled expression.
    pub fn exec_2in_2out(&mut self, in1: f64, in2: f64) -> (f64, f64, f64) {
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        let r = self.exec(in1, in2, 0.0, 0.0, 0.0, 0.0, &mut s1, &mut s2);
        (s1, s2, r)
    }

    /// Executes the machine code and provides the following parameters in order:
    /// `"in1", "in2", "alpha", "beta", "delta", "gamma", "&sig1", "&sig2"`
    ///
    /// It returns the return value of the computation. For addition outputs you can
    /// write to `"&sig1"` or `"&sig2"` with for instance: `assign(var("&sig1"), literal(10.0))`.
    #[allow(clippy::too_many_arguments)]
    pub fn exec(
        &mut self,
        in1: f64,
        in2: f64,
        alpha: f64,
        beta: f64,
        delta: f64,
        gamma: f64,
        sig1: &mut f64,
        sig2: &mut f64,
    ) -> f64 {
        {
            self.aux_vars[AUX_VAR_IDX_RESET] = if self.resetted {
                self.resetted = false;
                1.0
            } else {
                0.0
            };
        }
        let states_ptr: *const *mut u8 = self.node_states.as_mut_ptr();
        let pers_vars_ptr: *mut f64 = self.persistent_vars.as_mut_ptr();
        let aux_vars: *mut f64 = self.aux_vars.as_mut_ptr();
        let bufs: *const *mut f64 = unsafe { (*self.state).buffers.pointers().as_ptr() };
        let buf_lens: *const u64 = unsafe { (*self.state).buffers.lens().as_ptr() };
        let tables: *const *const f32 = unsafe { (*self.state).tables.pointers().as_ptr() };
        let table_lens: *const u64 = unsafe { (*self.state).tables.lens().as_ptr() };
        let mut multi_returns = [0.0; 5];

        (unsafe { self.function.unwrap_unchecked() })(
            in1,
            in2,
            alpha,
            beta,
            delta,
            gamma,
            sig1,
            sig2,
            aux_vars,
            self.state,
            states_ptr,
            pers_vars_ptr,
            (&mut multi_returns) as *mut f64,
            bufs,
            buf_lens,
            tables,
            table_lens,
        )
    }

    pub(crate) fn install(&mut self, node_state: &mut DSPNodeState) -> usize {
        let idx = self.node_states.len();
        node_state.mark(self.dsp_ctx_generation, idx);

        self.node_states.push(node_state.ptr());
        self.node_state_types.push(node_state.node_type());
        self.node_state_uids.push(node_state.uid());

        if !node_state.is_initialized() {
            self.node_state_init_reset.push(idx);
        }

        idx
    }

    pub(crate) fn touch_persistent_var_index(&mut self, idx: usize) {
        if idx >= self.persistent_vars.len() {
            self.persistent_vars.resize(idx + 1, 0.0);
        }
    }

    /// Gives you access to the persistent variables. To get the index of the
    /// persistent variable you must use [DSPNodeContext::get_persistent_variable_index_by_name].
    pub fn access_persistent_var(&mut self, idx: usize) -> Option<&mut f64> {
        self.persistent_vars.get_mut(idx)
    }

    /// Checks if the DSP function actually has the state for a certain unique DSP node state ID.
    pub fn has_dsp_node_state_uid(&self, uid: u64) -> bool {
        for i in self.node_state_uids.iter() {
            if *i == uid {
                return true;
            }
        }

        false
    }
}

impl Drop for DSPFunction {
    fn drop(&mut self) {
        unsafe {
            if let Some(module) = self.module.take() {
                module.free_memory();
            }
        };
    }
}

/// The global DSP state that all stateful [DSPNodeType] DSP nodes share.
pub struct DSPState {
    pub x: f64,
    pub y: f64,
    pub srate: f64,
    pub israte: f64,
    pub atoms: Vec<Arc<AtomicFloat>>,
    pub buffers: LockedMutPtrs<Vec<f64>, f64>,
    pub tables: LockedPtrs<Arc<Vec<f32>>, f32>,
}

/// An enum to specify the position of value and [DSPState] and [DSPNodeState] parameters
/// for the JIT compiler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DSPNodeSigBit {
    /// Signature placeholder for f64
    Value,
    /// Signature placeholder for the [DSPState] pointer
    DSPStatePtr,
    /// Signature placeholder for the [DSPNodeState] pointer that belongs to this node
    NodeStatePtr,
    /// Signature placeholder for a pointer to the multi return value array (max size is 5! `*mut [f64; 5]`)
    MultReturnPtr,
}

/// This trait allows you to define your own DSP stateful and stateless primitives.
/// Among defining a few important properties for the compiler, it handles allocation and
/// deallocation of the state that belongs to a DSPNodeType.
///
/// ## Stateless DSP Nodes/Primitives
///
/// Here is a simple example how to define a stateless DSP function:
///
///```
/// use std::rc::Rc;
/// use std::cell::RefCell;
/// use synfx_dsp_jit::{DSPNodeType, DSPNodeSigBit, DSPNodeTypeLibrary};
///
/// let lib = Rc::new(RefCell::new(DSPNodeTypeLibrary::new()));
///
/// pub struct MyPrimitive;
///
/// extern "C" fn my_primitive_function(a: f64, b: f64) -> f64 {
///    (2.0 * a * b.cos()).sin()
/// }
///
/// impl DSPNodeType for MyPrimitive {
///     // make a name, so you can refer to it via `ASTNode::Call("my_prim", ...)`.
///     fn name(&self) -> &str { "my_prim" }
///
///     // Provide a pointer:
///     fn function_ptr(&self) -> *const u8 { my_primitive_function as *const u8 }
///
///     // Define the function signature for the JIT compiler:
///     fn signature(&self, i: usize) -> Option<DSPNodeSigBit> {
///         match i {
///             0 | 1 => Some(DSPNodeSigBit::Value),
///             _ => None, // Return None to signal we only take 2 parameters
///         }
///     }
///
///     // Tell the JIT compiler that you return a value:
///     fn has_return_value(&self) -> bool { true }
///
///     // The other trait functions do not need to be provided, because this is
///     // a stateless primitive.
/// }
///
/// lib.borrow_mut().add(std::sync::Arc::new(MyPrimitive {}));
///
/// use synfx_dsp_jit::{ASTFun, JIT, DSPNodeContext};
/// let ctx = DSPNodeContext::new_ref();
/// let jit = JIT::new(lib.clone(), ctx.clone());
///
/// use synfx_dsp_jit::build::*;
/// let mut fun = jit.compile(ASTFun::new(
///     op_add(call("my_prim", 0, &[var("in1"), var("in2")]), literal(10.0))))
///     .expect("no compile error");
///
/// fun.init(44100.0, None);
///
/// let (_s1, _s2, ret) = fun.exec_2in_2out(1.0, 1.5);
///
/// assert!((ret - 10.1410029).abs() < 0.000001);
///
/// ctx.borrow_mut().free();
///```
///
/// ## Stateful DSP Nodes/Primitives
///
/// Here is a simple example how to define a stateful DSP function,
/// in this example just an accumulator.
///
/// There is a little helper macro that might help you: [crate::stateful_dsp_node_type]
///
///```
/// use std::rc::Rc;
/// use std::cell::RefCell;
/// use synfx_dsp_jit::{DSPNodeType, DSPState, DSPNodeSigBit, DSPNodeTypeLibrary};
///
/// let lib = Rc::new(RefCell::new(DSPNodeTypeLibrary::new()));
///
/// pub struct MyPrimitive;
///
/// struct MyPrimAccumulator {
///     count: f64,
/// }
///
/// // Be careful defining the signature of this primitive, there is no safety net here!
/// // Check twice with DSPNodeType::signature()!
/// extern "C" fn my_primitive_accum(add: f64, state: *mut u8) -> f64 {
///     let state = unsafe { &mut *(state as *mut MyPrimAccumulator) };
///     state.count += add;
///     state.count
/// }
///
/// impl DSPNodeType for MyPrimitive {
///     // make a name, so you can refer to it via `ASTNode::Call("my_prim", ...)`.
///     fn name(&self) -> &str { "accum" }
///
///     // Provide a pointer:
///     fn function_ptr(&self) -> *const u8 { my_primitive_accum as *const u8 }
///
///     // Define the function signature for the JIT compiler. Be really careful though,
///     // There is no safety net here.
///     fn signature(&self, i: usize) -> Option<DSPNodeSigBit> {
///         match i {
///             0 => Some(DSPNodeSigBit::Value),
///             1 => Some(DSPNodeSigBit::NodeStatePtr),
///             _ => None, // Return None to signal we only take 1 parameter
///         }
///     }
///
///     // Tell the JIT compiler that you return a value:
///     fn has_return_value(&self) -> bool { true }
///
///     // Specify how to reset the state:
///     fn reset_state(&self, _dsp_state: *mut DSPState, state_ptr: *mut u8) {
///         unsafe { (*(state_ptr as *mut MyPrimAccumulator)).count = 0.0 };
///     }
///
///     // Allocate our state:
///     fn allocate_state(&self) -> Option<*mut u8> {
///         Some(Box::into_raw(Box::new(MyPrimAccumulator { count: 0.0 })) as *mut u8)
///     }
///
///     // Deallocate our state:
///     fn deallocate_state(&self, ptr: *mut u8) {
///         unsafe { Box::from_raw(ptr as *mut MyPrimAccumulator) };
///     }
/// }
///
/// lib.borrow_mut().add(std::sync::Arc::new(MyPrimitive {}));
///
/// use synfx_dsp_jit::{ASTFun, JIT, DSPNodeContext};
/// let ctx = DSPNodeContext::new_ref();
/// let jit = JIT::new(lib.clone(), ctx.clone());
///
/// use synfx_dsp_jit::build::*;
/// let mut fun =
///     jit.compile(ASTFun::new(call("accum", 0, &[var("in1")]))).expect("no compile error");
///
/// fun.init(44100.0, None);
///
/// let (_s1, _s2, ret) = fun.exec_2in_2out(1.0, 0.0);
/// assert!((ret - 1.0).abs() < 0.000001);
///
/// let (_s1, _s2, ret) = fun.exec_2in_2out(1.0, 0.0);
/// assert!((ret - 2.0).abs() < 0.000001);
///
/// let (_s1, _s2, ret) = fun.exec_2in_2out(1.0, 0.0);
/// assert!((ret - 3.0).abs() < 0.000001);
///
/// // You can cause a reset eg. with fun.set_sample_rate() or fun.reset():
/// fun.reset();
///
/// // Counting will restart:
/// let (_s1, _s2, ret) = fun.exec_2in_2out(1.0, 0.0);
/// assert!((ret - 1.0).abs() < 0.000001);
///
/// ctx.borrow_mut().free();
///```
pub trait DSPNodeType: Sync + Send {
    /// The name of this DSP node, by this name it can be called from
    /// the [crate::ast::ASTFun].
    fn name(&self) -> &str;

    /// Document what this node does and how to use it.
    /// Format should be in Markdown.
    ///
    /// Documenting the node will make it easier for library implementors
    /// and even eventual end users to figure out what this node
    /// does and how to use it.
    ///
    /// For instance, this text should define what the input and output
    /// parameters do. And also define which value ranges these operate in.
    fn documentation(&self) -> &str {
        "undocumented"
    }

    /// Returns the name of each input port of this node.
    /// Choose descriptive but short names.
    /// These names will be used by compiler frontends to identify the ports,
    /// and it will make it easier to stay compatible if indices change.
    fn input_names(&self, _index: usize) -> Option<&str> {
        None
    }

    /// Returns the name of each output port of this node.
    /// Choose descriptive but short names.
    /// These names will be used by compiler frontends to identify the ports,
    /// and it will make it easier to stay compatible if indices change.
    fn output_names(&self, _index: usize) -> Option<&str> {
        None
    }

    /// Returns the index of the output by it's name.
    fn input_index_by_name(&self, name: &str) -> Option<usize> {
        let mut i = 0;

        while let Some(iname) = self.input_names(i) {
            if iname == name {
                return Some(i);
            }
            i += 1;
        }

        None
    }

    /// Returns the index of the output by it's name.
    fn output_index_by_name(&self, name: &str) -> Option<usize> {
        let mut i = 0;

        while let Some(oname) = self.output_names(i) {
            if oname == name {
                return Some(i);
            }
            i += 1;
        }

        None
    }

    /// Number of input ports
    fn input_count(&self) -> usize {
        let mut i = 0;
        while self.input_names(i).is_some() {
            i += 1;
        }
        i
    }

    /// Number of output ports
    fn output_count(&self) -> usize {
        let mut i = 0;
        while self.output_names(i).is_some() {
            i += 1;
        }
        i
    }

    /// Returns true if this node type requires state.
    fn is_stateful(&self) -> bool {
        let mut i = 0;
        while let Some(sig) = self.signature(i) {
            if let DSPNodeSigBit::NodeStatePtr = sig {
                return true;
            }

            i += 1;
        }

        false
    }

    /// The function pointer that should be inserted.
    fn function_ptr(&self) -> *const u8;

    /// Should return the signature type for input parameter `i`.
    fn signature(&self, _i: usize) -> Option<DSPNodeSigBit> {
        None
    }

    /// Should return true if the function for [DSPNodeType::function_ptr]
    /// returns something.
    fn has_return_value(&self) -> bool;

    /// Will be called when the node state should be resetted.
    /// This should be used to store the sample rate for instance or
    /// do other sample rate dependent recomputations.
    /// Also things delay lines should zero their buffers.
    fn reset_state(&self, _dsp_state: *mut DSPState, _state_ptr: *mut u8) {}

    /// Allocates a new piece of state for this [DSPNodeType].
    /// Must be deallocated using [DSPNodeType::deallocate_state].
    fn allocate_state(&self) -> Option<*mut u8> {
        None
    }

    /// Deallocates the private state of this [DSPNodeType].
    fn deallocate_state(&self, _ptr: *mut u8) {}
}

/// A handle to manage the state of a DSP node
/// that was created while the [crate::jit::DSPFunctionTranslator] compiled the given AST
/// to machine code. The AST needs to take care to refer to the same piece
/// of state with the same type across different compilations of the AST with the
/// same [DSPNodeContext].
///
/// It holds a pointer to the state of a single DSP node. The internal state
/// pointer will be shared with the execution thread that will execute the
/// complete DSP function/graph.
///
/// You will not have to allocate and manage this manually, see also [DSPFunction].
pub(crate) struct DSPNodeState {
    /// The node_state_uid that identifies this piece of state uniquely across multiple
    /// ASTs.
    uid: u64,
    /// Holds the type of this piece of state.
    node_type: Arc<dyn DSPNodeType>,
    /// A pointer to the allocated piece of state. It will be shared
    /// with the execution thread. So you must not touch the data that is referenced
    /// here.
    ptr: *mut u8,
    /// A generation counter that is used by [DSPNodeContext] to determine
    /// if a piece of state is not used anymore.
    generation: u64,
    /// The current index into the most recent [DSPFunction] that was
    /// constructed by [DSPNodeContext].
    function_index: usize,
    /// A flag that stores if this DSPNodeState instance was already initialized.
    /// It is set by [DSPNodeContext] if a finished [DSPFunction] was successfully compiled.
    initialized: bool,
}

impl DSPNodeState {
    /// Creates a fresh piece of DSP node state.
    pub(crate) fn new(uid: u64, node_type: Arc<dyn DSPNodeType>) -> Self {
        Self {
            uid,
            node_type: node_type.clone(),
            ptr: node_type.allocate_state().expect("DSPNodeState created for stateful node type"),
            generation: 0,
            function_index: 0,
            initialized: false,
        }
    }

    /// Returns the unique ID of this piece of DSP node state.
    pub(crate) fn uid(&self) -> u64 {
        self.uid
    }

    /// Marks this piece of DSP state as used and deposits the
    /// index into the current [DSPFunction].
    pub(crate) fn mark(&mut self, gen: u64, index: usize) {
        self.generation = gen;
        self.function_index = index;
    }

    /// Checks if the [DSPNodeState] was initialized by the most recently compiled [DSPFunction]
    pub(crate) fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Sets that the [DSPNodeState] is initialized.
    ///
    /// This happens once the [DSPNodeContext] finished compiling a [DSPFunction].
    /// The user of the [DSPNodeContext] or rather the [crate::JIT] needs to make sure to
    /// actually really call [DSPFunction::init] of course. Otherwise this state tracking
    /// all falls apart. But this happens across different threads, so the synchronizing effort
    /// for this is not worth it (regarding development time) at the moment I think.
    pub(crate) fn set_initialized(&mut self) {
        self.initialized = true;
    }

    /// Returns the state pointer for this DSPNodeState instance.
    /// Primarily used by [DSPFunction::install].
    pub(crate) fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Returns the [DSPNodeType] for this [DSPNodeState].
    pub(crate) fn node_type(&self) -> Arc<dyn DSPNodeType> {
        self.node_type.clone()
    }
}

impl Drop for DSPNodeState {
    /// This should only be dropped when the [DSPNodeContext] determined
    /// that the pointer that was shared with the execution thread is no longer
    /// in use.
    fn drop(&mut self) {
        self.node_type.deallocate_state(self.ptr);
        self.ptr = std::ptr::null_mut();
    }
}
