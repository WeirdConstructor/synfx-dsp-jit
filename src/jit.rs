// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

use crate::ast::*;
use cranelift::prelude::types::{F64, I32};
use cranelift::prelude::InstBuilder;
use cranelift::prelude::*;
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::default_libcall_names;
use cranelift_module::{FuncId, Linkage, Module};
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem;
use std::rc::Rc;

/// The basic JIT class.
pub struct JIT {
    /// The function builder context, which is reused across multiple
    /// FunctionBuilder instances.
    builder_context: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: codegen::Context,

    /// The module, with the jit backend, which manages the JIT'd
    /// functions.
    module: Option<JITModule>,

    /// The available DSP node types that an be called by the code.
    dsp_lib: Rc<RefCell<DSPNodeTypeLibrary>>,

    /// The current [DSPNodeContext] we compile a [DSPFunction] for
    dsp_ctx: Rc<RefCell<DSPNodeContext>>,
}

impl JIT {
    pub fn new(
        dsp_lib: Rc<RefCell<DSPNodeTypeLibrary>>,
        dsp_ctx: Rc<RefCell<DSPNodeContext>>,
    ) -> Self {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .expect("Setting 'use_colocated_libcalls' works");
        // FIXME set back to true once the x64 backend supports it.
        flag_builder.set("is_pic", "false").expect("Setting 'is_pic' works");
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .expect("ISA Builder finish works");
        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());

        dsp_lib
            .borrow()
            .for_each(|typ| -> Result<(), JITCompileError> {
                builder.symbol(typ.name(), typ.function_ptr());
                Ok(())
            })
            .expect("symbol adding works");

        let module = JITModule::new(builder);
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module: Some(module),
            dsp_lib,
            dsp_ctx,
        }
    }

    /// Compile a string in the toy language into machine code.
    pub fn compile(mut self, prog: ASTFun) -> Result<Box<DSPFunction>, JITCompileError> {
        let module = self.module.as_mut().expect("Module still loaded");
        let ptr_type = module.target_config().pointer_type();

        for param_idx in 0..prog.param_count() {
            if prog.param_is_ref(param_idx) {
                self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
            } else {
                self.ctx.func.signature.params.push(AbiParam::new(F64));
            };
        }

        self.ctx.func.signature.returns.push(AbiParam::new(F64));

        let id = module
            .declare_function("dsp", Linkage::Export, &self.ctx.func.signature)
            .map_err(|e| JITCompileError::DeclareTopFunError(e.to_string()))?;

        self.ctx.func.name = ExternalName::user(0, id.as_u32());

        // Then, translate the AST nodes into Cranelift IR.
        self.translate(prog)?;

        let mut module = self.module.take().expect("Module still loaded");
        module
            .define_function(id, &mut self.ctx)
            .map_err(|e| JITCompileError::DefineTopFunError(e.to_string()))?;

        module.clear_context(&mut self.ctx);
        module.finalize_definitions();

        let code = module.get_finalized_function(id);

        let dsp_fun = self
            .dsp_ctx
            .borrow_mut()
            .finalize_dsp_function(code, module)
            .expect("DSPFunction present in DSPNodeContext.");

        Ok(dsp_fun)
    }

    fn translate(&mut self, fun: ASTFun) -> Result<(), JITCompileError> {
        let builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        let module = self.module.as_mut().expect("Module still loaded");
        let dsp_lib = self.dsp_lib.clone();
        let dsp_lib = dsp_lib.borrow();
        let dsp_ctx = self.dsp_ctx.clone();
        let mut dsp_ctx = dsp_ctx.borrow_mut();
        let mut trans = DSPFunctionTranslator::new(&mut *dsp_ctx, &*dsp_lib, builder, module);
        trans.register_functions()?;
        let ret = trans.translate(fun)?;
        println!("{}", trans.builder.func.display());
        Ok(ret)
    }

    //    pub fn translate_ast_node(&mut self, builder: FunctionBuilder<'a>,
}

struct DSPFunctionTranslator<'a, 'b, 'c> {
    dsp_ctx: &'c mut DSPNodeContext,
    dsp_lib: &'b DSPNodeTypeLibrary,
    builder: FunctionBuilder<'a>,
    variables: HashMap<String, Variable>,
    var_index: usize,
    module: &'a mut JITModule,
    dsp_node_functions: HashMap<String, (Rc<dyn DSPNodeType>, FuncId)>,
    ptr_w: u32,
}

pub struct DSPState {
    pub x: f64,
    pub y: f64,
    pub srate: f64,
    pub israte: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DSPNodeSigBit {
    Value,
    DSPStatePtr,
    NodeStatePtr,
}

/// A trait that handles allocation and deallocation of the
/// state that belongs to a DSPNodeType.
pub trait DSPNodeType {
    /// The name of this DSP node, by this name it can be called from
    /// the [ASTFun].
    fn name(&self) -> &str;

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
/// that was created while the [DSPFunctionTranslator] compiled the given AST
/// to machine code. The AST needs to take care to refer to the same piece
/// of state with the same type across different compilations of the AST with the
/// same [DSPNodeContext].
///
/// It holds a pointer to the state of a single DSP node. The internal state
/// pointer will be shared with the execution thread that will execute the
/// complete DSP function/graph.
///
/// You will not have to allocate and manage this manually, see also [DSPFunction].
pub struct DSPNodeState {
    /// The node_state_uid that identifies this piece of state uniquely across multiple
    /// ASTs.
    uid: u64,
    /// Holds the type of this piece of state.
    node_type: Rc<dyn DSPNodeType>,
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
    pub fn new(uid: u64, node_type: Rc<dyn DSPNodeType>) -> Self {
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
    pub fn uid(&self) -> u64 {
        self.uid
    }

    /// Marks this piece of DSP state as used and deposits the
    /// index into the current [DSPFunction].
    pub fn mark(&mut self, gen: u64, index: usize) {
        self.generation = gen;
        self.function_index = index;
    }

    /// Checks if the [DSPNodeState] was initialized by the most recently compiled [DSPFunction]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Sets that the [DSPNodeState] is initialized.
    ///
    /// This happens once the [DSPNodeContext] finished compiling a [DSPFunction].
    /// The user of the [DSPNodeContext] or rather the [JIT] needs to make sure to
    /// actually really call [DSPFunction::init] of course. Otherwise this state tracking
    /// all falls apart. But this happens across different threads, so the synchronizing effort
    /// for this is not worth it (regarding development time) at the moment I think.
    pub fn set_initialized(&mut self) {
        self.initialized = true;
    }

    /// Returns the state pointer for this DSPNodeState instance.
    /// Primarily used by [DSPFunction::install].
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Returns the [DSPNodeType] for this [DSPNodeState].
    pub fn node_type(&self) -> Rc<dyn DSPNodeType> {
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

/// This structure holds all the [DSPNodeType] definitions and provides
/// them to the [JIT] and [DSPFunctionTranslator].
pub struct DSPNodeTypeLibrary {
    types: Vec<Rc<dyn DSPNodeType>>,
}

impl DSPNodeTypeLibrary {
    /// Create a new instance of this.
    pub fn new() -> Self {
        Self { types: vec![] }
    }

    /// Add the given [DSPNodeType] to this library.
    pub fn add(&mut self, typ: Rc<dyn DSPNodeType>) {
        self.types.push(typ);
    }

    /// Iterate through all types in the Library:
    ///
    ///```
    /// use wblockdsp::*;
    ///
    /// let lib = DSPNodeTypeLibrary::new();
    /// // ...
    /// lib.for_each(|typ| -> Result<(), ()> {
    ///     println!("Type available: {}", typ.name());
    ///     Ok(())
    /// }).expect("no error");
    ///```
    pub fn for_each<T, F: FnMut(&Rc<dyn DSPNodeType>) -> Result<(), T>>(
        &self,
        mut f: F,
    ) -> Result<(), T> {
        for t in self.types.iter() {
            f(&t)?;
        }
        Ok(())
    }
}

/// This table holds all the DSP state including the state of the individual DSP nodes
/// that were created by the [DSPFunctionTranslator].
pub struct DSPNodeContext {
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
}

impl DSPNodeContext {
    fn new() -> Self {
        Self {
            state: Box::into_raw(Box::new(DSPState {
                x: 0.0,
                y: 0.0,
                srate: 44100.0,
                israte: 1.0 / 44100.0,
            })),
            node_states: HashMap::new(),
            generation: 0,
            next_dsp_fun: None,
            persistent_var_map: HashMap::new(),
            persistent_var_index: 0,
        }
    }

    pub fn new_ref() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self::new()))
    }

    pub fn init_dsp_function(&mut self) {
        self.generation += 1;
        self.next_dsp_fun = Some(Box::new(DSPFunction::new(self.state, self.generation)));
    }

    /// Retrieve the index into the persistent variable vector passed in as "&pv".
    pub fn get_persistent_variable_index(&mut self, pers_var_name: &str) -> Result<usize, String> {
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

    /// Adds a [DSPNodeState] to the currently compiled [DSPFunction] and returns
    /// the index into the node state vector in the [DSPFunction], so that the JIT
    /// code can index into that vector to find the right state pointer.
    pub fn add_dsp_node_instance(
        &mut self,
        node_type: Rc<dyn DSPNodeType>,
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

    pub fn finalize_dsp_function(
        &mut self,
        function_ptr: *const u8,
        module: JITModule,
    ) -> Option<Box<DSPFunction>> {
        if let Some(mut next_dsp_fun) = self.next_dsp_fun.take() {
            next_dsp_fun.set_function_ptr(function_ptr, module);

            for (_, node_state) in self.node_states.iter_mut() {
                node_state.set_initialized();
            }

            Some(next_dsp_fun)
        } else {
            None
        }
    }

    pub fn cleanup_dsp_fun_after_user(&mut self, _fun: Box<DSPFunction>) {
        // TODO: Garbage collect and free unused node state!
        //       But this must happen by the backend/frontend thread separation.
        //       Best would be to provide DSPNodeContext::cleaup_dsp_function_after_use(DSPFunction).
    }

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

/// This is the result of the JIT compiled [ASTNode] tree.
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
    node_state_types: Vec<Rc<dyn DSPNodeType>>,
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
    function: Option<fn(
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        *mut f64,
        *mut f64,
        *mut DSPState,
        *mut *mut u8,
        *mut f64
    ) -> f64>,
}

unsafe impl Send for DSPFunction {}

impl DSPFunction {
    pub fn new(state: *mut DSPState, dsp_ctx_generation: u64) -> Self {
        Self {
            state,
            node_state_types: vec![],
            node_states: vec![],
            node_state_init_reset: vec![],
            node_state_uids: vec![],
            persistent_vars: vec![],
            function: None,
            dsp_ctx_generation,
            module: None,
        }
    }

    /// At the end of the compilation the [JIT] will put the resulting function
    /// pointer into this function.
    pub fn set_function_ptr(&mut self, function: *const u8, module: JITModule) {
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
                    f64,
                    f64,
                    *mut f64,
                    *mut f64,
                    *mut DSPState,
                    *mut *mut u8,
                    *mut f64,
                ) -> f64,
            >(function)
        });
    }

    pub fn init(&mut self, srate: f64, previous_function: Option<&DSPFunction>) {
        if let Some(previous_function) = previous_function {
            let prev_len = previous_function.persistent_vars.len();
            self.persistent_vars[0..prev_len]
                .copy_from_slice(&previous_function.persistent_vars[0..prev_len])
        }

        unsafe {
            (*self.state).srate = srate;
            (*self.state).israte = 1.0 / srate;
        }

        for idx in self.node_state_init_reset.iter() {
            let typ = &self.node_state_types[*idx as usize];
            let ptr = self.node_states[*idx as usize];
            typ.reset_state(self.state, ptr);
        }
    }

    pub fn set_sample_rate(&mut self, srate: f64) {
        unsafe {
            (*self.state).srate = srate;
            (*self.state).israte = 1.0 / srate;
        }

        self.reset();
    }

    pub fn reset(&mut self) {
        for (typ, ptr) in self.node_state_types.iter().zip(self.node_states.iter_mut()) {
            typ.reset_state(self.state, *ptr);
        }
    }

    pub fn get_dsp_state_ptr(&self) -> *mut DSPState {
        self.state
    }

    pub unsafe fn with_dsp_state<R, F: FnMut(*mut DSPState) -> R>(&mut self, mut f: F) -> R {
        f(self.get_dsp_state_ptr())
    }

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

    pub fn get_node_state_ptr(&self, node_state_uid: u64) -> Option<*mut u8> {
        for (i, uid) in self.node_state_uids.iter().enumerate() {
            if *uid == node_state_uid {
                return Some(self.node_states[i]);
            }
        }

        None
    }

    pub fn exec_2in_2out(&mut self, in1: f64, in2: f64) -> (f64, f64, f64) {
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        let r = self.exec(in1, in2, 0.0, 0.0, 0.0, 0.0, &mut s1, &mut s2);
        (s1, s2, r)
    }

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
        let (srate, israte) = unsafe { ((*self.state).srate, (*self.state).israte) };
        let states_ptr: *mut *mut u8 = self.node_states.as_mut_ptr();
        let pers_vars_ptr: *mut f64 = self.persistent_vars.as_mut_ptr();
        let ret = (unsafe { self.function.unwrap_unchecked() })(
            in1,
            in2,
            alpha,
            beta,
            delta,
            gamma,
            srate,
            israte,
            sig1,
            sig2,
            self.state,
            states_ptr,
            pers_vars_ptr,
        );
        ret
    }

    pub fn install(&mut self, node_state: &mut DSPNodeState) -> usize {
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

    pub fn touch_persistent_var_index(&mut self, idx: usize) {
        if idx >= self.persistent_vars.len() {
            self.persistent_vars.resize(idx + 1, 0.0);
        }
    }

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

#[derive(Debug, Clone)]
pub enum JITCompileError {
    BadDefinedParams,
    UnknownFunction(String),
    UndefinedVariable(String),
    DeclareTopFunError(String),
    DefineTopFunError(String),
    UndefinedDSPNode(String),
    NotEnoughArgsInCall(String, u64),
    NodeStateError(String, u64),
}

impl<'a, 'b, 'c> DSPFunctionTranslator<'a, 'b, 'c> {
    pub fn new(
        dsp_ctx: &'c mut DSPNodeContext,
        dsp_lib: &'b DSPNodeTypeLibrary,
        builder: FunctionBuilder<'a>,
        module: &'a mut JITModule,
    ) -> Self {
        dsp_ctx.init_dsp_function();

        Self {
            dsp_ctx,
            dsp_lib,
            var_index: 0,
            variables: HashMap::new(),
            builder,
            module,
            dsp_node_functions: HashMap::new(),
            ptr_w: 8,
        }
    }

    pub fn register_functions(&mut self) -> Result<(), JITCompileError> {
        let ptr_type = self.module.target_config().pointer_type();

        let mut dsp_node_functions = HashMap::new();
        self.dsp_lib.for_each(|typ| {
            let mut sig = self.module.make_signature();
            let mut i = 0;
            while let Some(bit) = typ.signature(i) {
                match bit {
                    DSPNodeSigBit::Value => {
                        sig.params.push(AbiParam::new(F64));
                    }
                    DSPNodeSigBit::DSPStatePtr | DSPNodeSigBit::NodeStatePtr => {
                        sig.params.push(AbiParam::new(ptr_type));
                    }
                }
                i += 1;
            }

            if typ.has_return_value() {
                sig.returns.push(AbiParam::new(F64));
            }

            let func_id = self
                .module
                .declare_function(typ.name(), cranelift_module::Linkage::Import, &sig)
                .map_err(|e| JITCompileError::DeclareTopFunError(e.to_string()))?;

            dsp_node_functions.insert(typ.name().to_string(), (typ.clone(), func_id));

            Ok(())
        })?;

        self.dsp_node_functions = dsp_node_functions;

        Ok(())
    }

    /// Declare a single variable declaration.
    fn declare_variable(&mut self, typ: types::Type, name: &str) -> Variable {
        let var = Variable::new(self.var_index);
        //d// println!("DECLARE {} = {}", name, self.var_index);

        if !self.variables.contains_key(name) {
            self.variables.insert(name.into(), var);
            self.builder.declare_var(var, typ);
            self.var_index += 1;
        }

        var
    }

    fn translate(&mut self, fun: ASTFun) -> Result<(), JITCompileError> {
        let ptr_type = self.module.target_config().pointer_type();
        self.ptr_w = ptr_type.bytes();

        let entry_block = self.builder.create_block();
        self.builder.append_block_params_for_function_params(entry_block);
        self.builder.switch_to_block(entry_block);
        self.builder.seal_block(entry_block);

        self.variables.clear();

        // declare and define parameters:
        for param_idx in 0..fun.param_count() {
            let val = self.builder.block_params(entry_block)[param_idx];

            match fun.param_name(param_idx) {
                Some(param_name) => {
                    let var = if fun.param_is_ref(param_idx) {
                        self.declare_variable(ptr_type, param_name)
                    } else {
                        self.declare_variable(F64, param_name)
                    };

                    self.builder.def_var(var, val);
                }
                None => {
                    return Err(JITCompileError::BadDefinedParams);
                }
            }
        }

        // declare and define local variables:
        for local_name in fun.local_variables().iter() {
            let zero = self.builder.ins().f64const(0.0);
            let var = self.declare_variable(F64, local_name);
            self.builder.def_var(var, zero);
        }

        let v = self.compile(fun.ast_ref())?;

        self.builder.ins().return_(&[v]);
        self.builder.finalize();

        Ok(())
    }

    fn ins_b_to_f64(&mut self, v: Value) -> Value {
        let bint = self.builder.ins().bint(I32, v);
        self.builder.ins().fcvt_from_uint(F64, bint)
    }

    fn compile(&mut self, ast: &Box<ASTNode>) -> Result<Value, JITCompileError> {
        match ast.as_ref() {
            ASTNode::Lit(v) => Ok(self.builder.ins().f64const(*v)),
            ASTNode::Var(name) => {
                if name.chars().next() == Some('&') {
                    let variable = self
                        .variables
                        .get(name)
                        .ok_or_else(|| JITCompileError::UndefinedVariable(name.to_string()))?;
                    let ptr = self.builder.use_var(*variable);
                    Ok(self.builder.ins().load(F64, MemFlags::new(), ptr, 0))
                } else if name.chars().next() == Some('*') {
                    let pv_index = self
                        .dsp_ctx
                        .get_persistent_variable_index(name)
                        .or_else(|_| Err(JITCompileError::UndefinedVariable(name.to_string())))?;

                    let persistent_vars = self
                        .variables
                        .get("&pv")
                        .ok_or_else(|| JITCompileError::UndefinedVariable("&pv".to_string()))?;
                    let pvs = self.builder.use_var(*persistent_vars);
                    let pers_value = self.builder.ins().load(
                        F64,
                        MemFlags::new(),
                        pvs,
                        Offset32::new(pv_index as i32 * F64.bytes() as i32),
                    );
                    Ok(pers_value)
                } else {
                    let variable = self
                        .variables
                        .get(name)
                        .ok_or_else(|| JITCompileError::UndefinedVariable(name.to_string()))?;
                    Ok(self.builder.use_var(*variable))
                }
            }
            ASTNode::Assign(name, ast) => {
                let value = self.compile(ast)?;

                if name.chars().next() == Some('&') {
                    let variable = self
                        .variables
                        .get(name)
                        .ok_or_else(|| JITCompileError::UndefinedVariable(name.to_string()))?;
                    let ptr = self.builder.use_var(*variable);
                    self.builder.ins().store(MemFlags::new(), value, ptr, 0);

                } else if name.chars().next() == Some('*') {
                    let pv_index = self
                        .dsp_ctx
                        .get_persistent_variable_index(name)
                        .or_else(|_| Err(JITCompileError::UndefinedVariable(name.to_string())))?;

                    let persistent_vars = self
                        .variables
                        .get("&pv")
                        .ok_or_else(|| JITCompileError::UndefinedVariable("&pv".to_string()))?;
                    let pvs = self.builder.use_var(*persistent_vars);
                    self.builder.ins().store(
                        MemFlags::new(),
                        value,
                        pvs,
                        Offset32::new(pv_index as i32 * F64.bytes() as i32),
                    );

                } else {
                    let variable = self
                        .variables
                        .get(name)
                        .ok_or_else(|| JITCompileError::UndefinedVariable(name.to_string()))?;
                    self.builder.def_var(*variable, value);
                }

                Ok(value)
            }
            ASTNode::BinOp(op, a, b) => {
                let value_a = self.compile(a)?;
                let value_b = self.compile(b)?;
                let value = match op {
                    ASTBinOp::Add => self.builder.ins().fadd(value_a, value_b),
                    ASTBinOp::Sub => self.builder.ins().fsub(value_a, value_b),
                    ASTBinOp::Mul => self.builder.ins().fmul(value_a, value_b),
                    ASTBinOp::Div => self.builder.ins().fdiv(value_a, value_b),
                    ASTBinOp::Eq => {
                        let cmp_res = self.builder.ins().fcmp(FloatCC::Equal, value_a, value_b);
                        self.ins_b_to_f64(cmp_res)
                    }
                    ASTBinOp::Ne => {
                        let cmp_res = self.builder.ins().fcmp(FloatCC::Equal, value_a, value_b);
                        let bnot = self.builder.ins().bnot(cmp_res);
                        let bint = self.builder.ins().bint(I32, bnot);
                        self.builder.ins().fcvt_from_uint(F64, bint)
                    }
                    ASTBinOp::Ge => {
                        let cmp_res =
                            self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, value_a, value_b);
                        self.ins_b_to_f64(cmp_res)
                    }
                    ASTBinOp::Le => {
                        let cmp_res =
                            self.builder.ins().fcmp(FloatCC::LessThanOrEqual, value_a, value_b);
                        self.ins_b_to_f64(cmp_res)
                    }
                    ASTBinOp::Gt => {
                        let cmp_res =
                            self.builder.ins().fcmp(FloatCC::GreaterThan, value_a, value_b);
                        self.ins_b_to_f64(cmp_res)
                    }
                    ASTBinOp::Lt => {
                        let cmp_res = self.builder.ins().fcmp(FloatCC::LessThan, value_a, value_b);
                        self.ins_b_to_f64(cmp_res)
                    }
                };

                Ok(value)
            }
            ASTNode::Call(name, dsp_node_uid, args) => {
                let func = self
                    .dsp_node_functions
                    .get(name)
                    .ok_or_else(|| JITCompileError::UndefinedDSPNode(name.to_string()))?
                    .clone();
                let node_type = func.0;
                let func_id = func.1;

                let ptr_type = self.module.target_config().pointer_type();

                let mut dsp_node_fun_params = vec![];
                let mut i = 0;
                let mut arg_idx = 0;
                while let Some(bit) = node_type.signature(i) {
                    match bit {
                        DSPNodeSigBit::Value => {
                            if arg_idx >= args.len() {
                                return Err(JITCompileError::NotEnoughArgsInCall(
                                    name.to_string(),
                                    *dsp_node_uid,
                                ));
                            }
                            dsp_node_fun_params.push(self.compile(&args[arg_idx])?);
                            arg_idx += 1;
                        }
                        DSPNodeSigBit::DSPStatePtr => {
                            let state_var = self.variables.get("&state").ok_or_else(|| {
                                JITCompileError::UndefinedVariable("&state".to_string())
                            })?;
                            dsp_node_fun_params.push(self.builder.use_var(*state_var));
                        }
                        DSPNodeSigBit::NodeStatePtr => {
                            let node_state_index = match self
                                .dsp_ctx
                                .add_dsp_node_instance(node_type.clone(), *dsp_node_uid)
                            {
                                Err(e) => {
                                    return Err(JITCompileError::NodeStateError(e, *dsp_node_uid));
                                }
                                Ok(idx) => idx,
                            };

                            let fstate_var = self.variables.get("&fstate").ok_or_else(|| {
                                JITCompileError::UndefinedVariable("&fstate".to_string())
                            })?;
                            let fptr = self.builder.use_var(*fstate_var);
                            let func_state = self.builder.ins().load(
                                ptr_type,
                                MemFlags::new(),
                                fptr,
                                Offset32::new(node_state_index as i32 * self.ptr_w as i32),
                            );
                            dsp_node_fun_params.push(func_state);
                        }
                    }

                    i += 1;
                }

                let local_callee =
                    self.module.declare_func_in_func(func_id, &mut self.builder.func);
                let call = self.builder.ins().call(local_callee, &dsp_node_fun_params);
                Ok(self.builder.inst_results(call)[0])
            }
            ASTNode::If(cond, then, els) => {
                let condition_value = if let ASTNode::BinOp(op, a, b) = cond.as_ref() {
                    let val = match op {
                        ASTBinOp::Eq => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            self.builder.ins().fcmp(FloatCC::Equal, a, b)
                        }
                        ASTBinOp::Ne => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            let eq = self.builder.ins().fcmp(FloatCC::Equal, a, b);
                            self.builder.ins().bnot(eq)
                        }
                        ASTBinOp::Gt => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            self.builder.ins().fcmp(FloatCC::GreaterThan, a, b)
                        }
                        ASTBinOp::Lt => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            self.builder.ins().fcmp(FloatCC::LessThan, a, b)
                        }
                        ASTBinOp::Ge => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, a, b)
                        }
                        ASTBinOp::Le => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            self.builder.ins().fcmp(FloatCC::LessThanOrEqual, a, b)
                        }
                        _ => self.compile(cond)?,
                    };

                    val
                } else {
                    let res = self.compile(cond)?;
                    let cmpv = self.builder.ins().f64const(0.5);
                    self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, res, cmpv)
                };

                let then_block = self.builder.create_block();
                let else_block = self.builder.create_block();
                let merge_block = self.builder.create_block();

                // If-else constructs in the toy language have a return value.
                // In traditional SSA form, this would produce a PHI between
                // the then and else bodies. Cranelift uses block parameters,
                // so set up a parameter in the merge block, and we'll pass
                // the return values to it from the branches.
                self.builder.append_block_param(merge_block, F64);

                // Test the if condition and conditionally branch.
                self.builder.ins().brz(condition_value, else_block, &[]);
                // Fall through to then block.
                self.builder.ins().jump(then_block, &[]);

                self.builder.switch_to_block(then_block);
                self.builder.seal_block(then_block);
                let then_return = self.compile(then)?;

                // Jump to the merge block, passing it the block return value.
                self.builder.ins().jump(merge_block, &[then_return]);

                self.builder.switch_to_block(else_block);
                self.builder.seal_block(else_block);
                let else_return = if let Some(els) = els {
                    self.compile(els)?
                } else {
                    self.builder.ins().f64const(0.0)
                };

                // Jump to the merge block, passing it the block return value.
                self.builder.ins().jump(merge_block, &[else_return]);

                // Switch to the merge block for subsequent statements.
                self.builder.switch_to_block(merge_block);

                // We've now seen all the predecessors of the merge block.
                self.builder.seal_block(merge_block);

                // Read the value of the if-else by reading the merge block
                // parameter.
                let phi = self.builder.block_params(merge_block)[0];

                Ok(phi)
            }
            ASTNode::Stmts(stmts) => {
                let mut value = None;
                for ast in stmts {
                    value = Some(self.compile(ast)?);
                }
                if let Some(value) = value {
                    Ok(value)
                } else {
                    Ok(self.builder.ins().f64const(0.0))
                }
            }
        }
    }
}

#[macro_export]
macro_rules! stateful_dsp_node_type {
    ($node_type: ident, $struct_type: ident =>
        $func_name: ident $jit_name: literal $signature: literal) => {
        struct $node_type;
        impl $node_type {
            fn new_ref() -> std::rc::Rc<Self> {
                std::rc::Rc::new(Self {})
            }
        }
        impl wblockdsp::DSPNodeType for $node_type {
            fn name(&self) -> &str {
                $jit_name
            }

            fn function_ptr(&self) -> *const u8 {
                $func_name as *const u8
            }

            fn signature(&self, i: usize) -> Option<DSPNodeSigBit> {
                match $signature.chars().nth(i).unwrap() {
                    'v' => Some(DSPNodeSigBit::Value),
                    'D' => Some(DSPNodeSigBit::DSPStatePtr),
                    'S' => Some(DSPNodeSigBit::NodeStatePtr),
                    _ => None,
                }
            }

            fn has_return_value(&self) -> bool {
                $signature.find("r").is_some()
            }

            fn reset_state(&self, dsp_state: *mut wblockdsp::DSPState, state_ptr: *mut u8) {
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
        }
    };
}

pub fn get_nop_function(
    lib: Rc<RefCell<DSPNodeTypeLibrary>>,
    dsp_ctx: Rc<RefCell<DSPNodeContext>>,
) -> Box<DSPFunction> {
    let jit = JIT::new(lib, dsp_ctx);
    jit.compile(ASTFun::new(Box::new(ASTNode::Lit(0.0)))).expect("No compile error")
}
