// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

use cranelift_jit::JITModule;
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem;
use std::rc::Rc;

/// This table holds all the DSP state including the state of the individual DSP nodes
/// that were created by the [crate::jit::DSPFunctionTranslator].
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

    pub(crate) fn init_dsp_function(&mut self) {
        self.generation += 1;
        self.next_dsp_fun = Some(Box::new(DSPFunction::new(self.state, self.generation)));
    }

    /// Retrieve the index into the persistent variable vector passed in as "&pv".
    pub(crate) fn get_persistent_variable_index(&mut self, pers_var_name: &str) -> Result<usize, String> {
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
    pub(crate) fn add_dsp_node_instance(
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

    pub(crate) fn finalize_dsp_function(
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

/// This structure holds all the [DSPNodeType] definitions and provides
/// them to the [crate::JIT] and [crate::jit::DSPFunctionTranslator].
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
    /// use synfx_dsp_jit::*;
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
    function: Option<
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
    >,
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

    /// At the end of the compilation the [crate::JIT] will put the resulting function
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

/// The global DSP state that all stateful [DSPNodeType] DSP nodes share.
pub struct DSPState {
    pub x: f64,
    pub y: f64,
    pub srate: f64,
    pub israte: f64,
}

/// An enum to specify the position of value and [DSPState] and [DSPNodeState] parameters
/// for the JIT compiler.
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
    /// the [crate::ast::ASTFun].
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
    pub(crate) fn new(uid: u64, node_type: Rc<dyn DSPNodeType>) -> Self {
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
    pub(crate) fn node_type(&self) -> Rc<dyn DSPNodeType> {
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
