// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

/*! This module implements a real time capable engine for sharing [DSPFunction]s with an audio thread.

Use this if you plan on (re)compiling little DSP functions in
a frontend thread and having an audio/backend thread actually executing the [DSPFunction].

The [CodeEngine] handles allocation and deallocation for you, as well as communication
with the real time thread.

See also [CodeEngine] or [crate] for API examples. There is also an example included
with this crate.
*/

use crate::*;

use ringbuf::{Consumer, Producer, RingBuffer};
use synfx_dsp::AtomicFloat;
use std::sync::Arc;
use std::cell::RefCell;
use std::rc::Rc;

const MAX_RINGBUF_SIZE: usize = 128;

enum CodeUpdateMsg {
    UpdateFun(Box<DSPFunction>),
    ResetFun,
}

enum CodeReturnMsg {
    DestroyFun(Box<DSPFunction>),
}

/// This is the frontend handle for the DSP code execution engine.
///
/// You create it with either [CodeEngine::new_with_lib] or [CodeEngine::new_stdlib].
/// Afterwards you split off the backend/real time thread handle with [CodeEngine::get_backend].
/// In the backend you must make sure to call [CodeEngineBackend::set_sample_rate] at least once
/// and [CodeEngineBackend::process_updates] regularily.
///
/// Once the audio thread runs, you can call [CodeEngine::upload] with an [ASTNode] tree.
/// The tree can be built for instance with the helper functions in [crate::build].
/// To process feedback and unused old [DSPFunction] instances, you **must** call
/// [CodeEngine::query_returns] regularily. In a GUI for instance each frame, or in the idle callback
/// of the event loop.
///
/// This is the rough way to use this API:
///
///```
/// use synfx_dsp_jit::engine::CodeEngine;
/// use synfx_dsp_jit::build::*;
///
/// let mut engine = CodeEngine::new_stdlib();
///
/// let mut backend = engine.get_backend();
/// std::thread::spawn(move || {
///     backend.set_sample_rate(44100.0);
///
///     loop {
///         backend.process_updates();
///
///         for frame in 0..64 {
///             let (s1, s2, ret) = backend.process(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
///         }
///     }
/// });
///
/// // Upload a new piece of code:
/// engine.upload(call("sin", 1, &[literal(1.0)])).unwrap();
///
/// // Call this regularily!!!!
/// engine.query_returns();
///```
///
/// A more elaborate example can be found at the top level: [crate]
pub struct CodeEngine {
    dsp_ctx: Rc<RefCell<DSPNodeContext>>,
    lib: Rc<RefCell<DSPNodeTypeLibrary>>,
    update_prod: Producer<CodeUpdateMsg>,
    return_cons: Consumer<CodeReturnMsg>,
    ast_dump: String,
}

impl Clone for CodeEngine {
    fn clone(&self) -> Self {
        CodeEngine::new_with_lib(self.lib.clone())
    }
}

impl CodeEngine {
    /// Constructor for your custom [DSPNodeTypeLibrary].
    pub fn new_with_lib(lib: Rc<RefCell<DSPNodeTypeLibrary>>) -> Self {
        let rb = RingBuffer::new(MAX_RINGBUF_SIZE);
        let (update_prod, _update_cons) = rb.split();
        let rb = RingBuffer::new(MAX_RINGBUF_SIZE);
        let (_return_prod, return_cons) = rb.split();

        Self {
            lib,
            dsp_ctx: DSPNodeContext::new_ref(),
            update_prod,
            return_cons,
            ast_dump: String::from(""),
        }
    }

    /// Enabled debug information collection:
    pub fn set_debug(&mut self, debug: bool) {
        self.dsp_ctx.borrow_mut().set_debug(debug);
    }

    /// Retrieves debug information:
    pub fn get_debug_info(&self) -> String {
        format!("---------- AST ----------\n{}\n------- JIT IR -------\n{}\n---------- END ----------",
            self.ast_dump,
            self.dsp_ctx.borrow_mut().get_ir_dump()
        )
    }

    /// Constructor with the default standard library that comes with `synfx-dsp-jit`.
    pub fn new_stdlib() -> Self {
        Self::new_with_lib(get_standard_library())
    }

    /// Returns the [DSPNodeTypeLibrary] that is used by this [CodeEngine].
    pub fn get_lib(&self) -> Rc<RefCell<DSPNodeTypeLibrary>> {
        self.lib.clone()
    }

    /// Returns you a reference to the specified atom connected with the DSP backend.
    /// These atoms can be read and written in the [DSPFunction] using the `atomr` and `atomw`
    /// nodes.
    pub fn atom(&self, idx: usize) -> Option<Arc<AtomicFloat>> {
        self.dsp_ctx.borrow().atom(idx)
    }

    /// A shortcut to access a specific atom that was written with the `atomw` node.
    /// An alternative is the [CodeEngine::atom] method to directly access the [AtomicFloat].
    pub fn atom_get(&self, idx: usize) -> f32 {
        self.atom(idx).map(|a| a.get()).unwrap_or(0.0)
    }

    /// A shortcut to access a specific atom that can be read with the `atomr` (or `atomr~`) node.
    /// An alternative is the [CodeEngine::atom] method to directly access the [AtomicFloat].
    pub fn atom_set(&self, idx: usize, v: f32) {
        if let Some(at) = self.atom(idx) {
            at.set(v)
        }
    }

    /// Compiles and uploads a new piece of DSP code to the backend thread.
    ///
    ///```
    /// use synfx_dsp_jit::engine::CodeEngine;
    /// use synfx_dsp_jit::build::*;
    ///
    /// let mut engine = CodeEngine::new_stdlib();
    /// // ..
    /// engine.upload(call("sin", 1, &[literal(1.0)])).unwrap();
    /// // ..
    ///```
    pub fn upload(&mut self, ast: Box<ASTNode>) -> Result<(), JITCompileError> {
        let jit = JIT::new(self.lib.clone(), self.dsp_ctx.clone());
        if self.dsp_ctx.borrow().debug_enabled() {
            self.ast_dump = ast.dump(0);
        }
        let fun = jit.compile(ASTFun::new(ast))?;
        let _ = self.update_prod.push(CodeUpdateMsg::UpdateFun(fun));

        Ok(())
    }

    pub fn send_buffer(&mut self, index: usize, buf: Vec<f64>) {
    }

    pub fn send_table(&mut self, index: usize, buf: Arc<Vec<f64>>) {
    }

    /// Emits a message to the backend to cause a reset of the DSPFunction.
    /// All non persistent state is resetted in this case.
    pub fn reset(&mut self) {
        let _ = self.update_prod.push(CodeUpdateMsg::ResetFun);
    }

    fn cleanup(&self, fun: Box<DSPFunction>) {
        self.dsp_ctx.borrow_mut().cleanup_dsp_fun_after_user(fun);
    }

    /// Call this regularily in the frontend/worker thread for cleanup purposes.
    pub fn query_returns(&mut self) {
        while let Some(msg) = self.return_cons.pop() {
            match msg {
                CodeReturnMsg::DestroyFun(fun) => {
                    self.cleanup(fun);
                }
            }
        }
    }

    /// Use this function to split off a [CodeEngineBackend]
    /// handle. If you call this multiple times, the previously generated
    /// [CodeEngineBackend] instances will not receive any updates anymore
    /// (for now).
    ///
    ///```
    /// use synfx_dsp_jit::engine::CodeEngine;
    /// use synfx_dsp_jit::build::*;
    ///
    /// let mut engine = CodeEngine::new_stdlib();
    ///
    /// let mut backend = engine.get_backend();
    /// std::thread::spawn(move || {
    ///     backend.set_sample_rate(44100.0);
    ///     loop {
    ///         backend.process_updates();
    ///         // ...
    ///     }
    /// });
    ///```
    /// See also the module description for a more complete example [crate::engine]
    pub fn get_backend(&mut self) -> CodeEngineBackend {
        let rb = RingBuffer::new(MAX_RINGBUF_SIZE);
        let (update_prod, update_cons) = rb.split();
        let rb = RingBuffer::new(MAX_RINGBUF_SIZE);
        let (return_prod, return_cons) = rb.split();

        self.update_prod = update_prod;
        self.return_cons = return_cons;

        let function = get_nop_function(self.lib.clone(), self.dsp_ctx.clone());
        CodeEngineBackend::new(function, update_cons, return_prod)
    }
}

impl Drop for CodeEngine {
    fn drop(&mut self) {
        self.dsp_ctx.borrow_mut().free();
    }
}

/// The backend handle for a [CodeEngine].
///
/// You get this from a call to [CodeEngine::get_backend].
/// Make sure to set it up properly with [CodeEngineBackend::set_sample_rate] and
/// regularily call [CodeEngineBackend::process_updates] for receiving updated
/// [DSPFunction] instances from [CodeEngine::upload].
pub struct CodeEngineBackend {
    sample_rate: f32,
    function: Box<DSPFunction>,
    update_cons: Consumer<CodeUpdateMsg>,
    return_prod: Producer<CodeReturnMsg>,
}

impl CodeEngineBackend {
    fn new(
        function: Box<DSPFunction>,
        update_cons: Consumer<CodeUpdateMsg>,
        return_prod: Producer<CodeReturnMsg>,
    ) -> Self {
        Self { sample_rate: 0.0, function, update_cons, return_prod }
    }

    #[inline]
    pub fn process(
        &mut self,
        in1: f32,
        in2: f32,
        a: f32,
        b: f32,
        d: f32,
        g: f32,
    ) -> (f32, f32, f32) {
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        let res = self
            .function
            .exec(in1 as f64, in2 as f64, a as f64, b as f64, d as f64, g as f64, &mut s1, &mut s2);
        (s1 as f32, s2 as f32, res as f32)
    }

    /// Update/set the sample rate for the DSP function.
    /// This will also reset the state of the DSP function.
    pub fn set_sample_rate(&mut self, srate: f32) {
        self.sample_rate = srate;
        self.function.set_sample_rate(srate as f64);
    }

    /// Reset the state of the DSP function.
    pub fn clear(&mut self) {
        self.function.reset();
    }

    /// Process updates received from the thread running the [CodeEngine].
    pub fn process_updates(&mut self) {
        while let Some(msg) = self.update_cons.pop() {
            match msg {
                CodeUpdateMsg::UpdateFun(mut fun) => {
                    std::mem::swap(&mut self.function, &mut fun);
                    self.function.init(self.sample_rate as f64, Some(&fun));
                    let _ = self.return_prod.push(CodeReturnMsg::DestroyFun(fun));
                }
                CodeUpdateMsg::ResetFun => {
            println!("UPRESET");
                    self.function.reset();
                },
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn check_engine_reset() {
        use crate::build::*;

        let mut engine = CodeEngine::new_stdlib();
        let mut backend = engine.get_backend();

        backend.set_sample_rate(44100.0);

        engine.upload(var("$reset")).unwrap();

        backend.process_updates();
        let (_s1, _s2, ret) = backend.process(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(ret.round() as i32, 0);

        engine.reset();
        backend.process_updates();
        let (_s1, _s2, ret) = backend.process(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(ret.round() as i32, 1);
    }

    #[test]
    fn check_engine_buffer_size_change() {
        use crate::build::*;

        let mut engine = CodeEngine::new_stdlib();
        let mut backend = engine.get_backend();

        backend.set_sample_rate(44100.0);

        engine.upload(op_add(buf_len(0), buf_len(1))).unwrap();

        backend.process_updates();
        let (_s1, _s2, ret) = backend.process(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(ret.round() as i32, 32);

        engine.upload(stmts(&[buf_declare(0, 1), buf_declare(1, 256)])).unwrap();
        backend.process_updates();

        engine.upload(op_add(buf_len(0), buf_len(1))).unwrap();
        backend.process_updates();
        let (_s1, _s2, ret) = backend.process(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(ret.round() as i32, 257);
    }

    #[test]
    fn check_engine_buffer_table_sending() {
        use crate::build::*;

        let mut engine = CodeEngine::new_stdlib();
        let mut backend = engine.get_backend();

        backend.set_sample_rate(44100.0);

        engine.upload(
            stmts(&[
                assign("&sig1", buf_read(4, literal(5.0))),
                assign("&sig2", table_read(3, literal(5.0))),
                op_add(buf_len(4), table_len(3)),
            ])
        ).unwrap();

        backend.process_updates();
        let (s1, s2, ret) = backend.process(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(s1, 0.0);
        assert_eq!(s2, 0.0);
        assert_eq!(ret.round() as i32, 32);

        engine.send_buffer(4, vec![0.4532; 10]);
        engine.send_table(3, std::sync::Arc::new(vec![0.5532; 23]));
        backend.process_updates();

        let (s1, s2, ret) = backend.process(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(s1, 0.45);
        assert_eq!(s2, 0.55);
        assert_eq!(ret.round() as i32, 33);
    }
}
