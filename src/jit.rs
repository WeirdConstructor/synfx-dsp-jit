// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

use crate::ast::{ASTBinOp, ASTBufOp, ASTFun, ASTNode, ASTLenOp};
use crate::context::{
    DSPFunction, DSPNodeContext, DSPNodeSigBit, DSPNodeType, DSPNodeTypeLibrary,
    AUX_VAR_IDX_ISRATE, AUX_VAR_IDX_RESET, AUX_VAR_IDX_SRATE,
};
use cranelift::prelude::types::{F64, F32, I32, I64};
use cranelift::prelude::InstBuilder;
use cranelift::prelude::*;
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::UserFuncName;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::default_libcall_names;
use cranelift_module::{FuncId, Linkage, Module};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

/// The Just In Time compiler, that translates a [crate::ASTNode] tree into
/// machine code in form of a [DSPFunction] structure you can use to execute it.
///
/// See also [JIT::compile] for an example.
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
    /// Create a new JIT compiler instance.
    ///
    /// Because every newly compile function gets it's own fresh module,
    /// you need to recreate a [JIT] instance for every time you compile
    /// a function.
    ///
    ///```
    /// use synfx_dsp_jit::*;
    /// let lib = get_standard_library();
    /// let ctx = DSPNodeContext::new_ref();
    ///
    /// let jit = JIT::new(lib.clone(), ctx.clone());
    /// // ...
    /// ctx.borrow_mut().free();
    ///```
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

    /// Compiles a [crate::ASTFun] / [crate::ASTNode] tree into a [DSPFunction].
    ///
    /// There are some checks done by the compiler, see the possible errors in [JITCompileError].
    /// Otherwise the usage is pretty straight forward, here is another example:
    ///```
    /// use synfx_dsp_jit::*;
    /// let lib = get_standard_library();
    /// let ctx = DSPNodeContext::new_ref();
    ///
    /// let jit = JIT::new(lib.clone(), ctx.clone());
    /// let mut fun = jit.compile(ASTFun::new(Box::new(ASTNode::Lit(0.424242))))
    ///     .expect("Compiles fine");
    ///
    /// // ...
    /// fun.init(44100.0, None);
    /// // ...
    /// let (mut sig1, mut sig2) = (0.0, 0.0);
    /// let ret = fun.exec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &mut sig1, &mut sig2);
    /// // ...
    ///
    /// // Compile a different function now...
    /// let jit = JIT::new(lib.clone(), ctx.clone());
    /// let mut new_fun = jit.compile(ASTFun::new(Box::new(ASTNode::Lit(0.33333))))
    ///     .expect("Compiles fine");
    ///
    /// // Make sure to preserve any (possible) state...
    /// new_fun.init(44100.0, Some(&fun));
    /// // ...
    /// let (mut sig1, mut sig2) = (0.0, 0.0);
    /// let ret = new_fun.exec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &mut sig1, &mut sig2);
    /// // ...
    ///
    /// ctx.borrow_mut().free();
    ///```
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

        self.ctx.func.name = UserFuncName::user(0, id.as_u32());

        // Then, translate the AST nodes into Cranelift IR.
        self.translate(prog)?;

        let mut module = self.module.take().expect("Module still loaded");
        module.define_function(id, &mut self.ctx).map_err(|e| {
            match e {
                cranelift_module::ModuleError::Compilation(e) => {
                    JITCompileError::DefineTopFunError(cranelift_codegen::print_errors::pretty_error(
                        &self.ctx.func,
                        e,
                    ))
                },
                _ => {
                    JITCompileError::DefineTopFunError(format!("{:?}", e))
                }
            }
        })?;

        module.clear_context(&mut self.ctx);
        match module.finalize_definitions() {
            Ok(()) => (),
            Err(e) => {
                return Err(JITCompileError::ModuleError(
                    format!("{}", e)));
            }
        }

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

        let debug = dsp_ctx.debug_enabled();

        let debug_str = {
            let mut trans = DSPFunctionTranslator::new(&mut *dsp_ctx, &*dsp_lib, builder, module);
            trans.register_functions()?;
            trans.translate(fun, debug)?
        };

        if let Some(debug_str) = debug_str {
            dsp_ctx.cranelift_ir_dump = debug_str;
        }

        Ok(())
    }

    //    pub fn translate_ast_node(&mut self, builder: FunctionBuilder<'a>,
}

fn constant_lookup(name: &str) -> Option<f64> {
    match name {
        "PI" => Some(std::f64::consts::PI),
        "TAU" => Some(std::f64::consts::TAU),
        "E" => Some(std::f64::consts::E),
        "1PI" => Some(std::f64::consts::FRAC_1_PI),
        "2PI" => Some(std::f64::consts::FRAC_2_PI),
        "PI2" => Some(std::f64::consts::FRAC_PI_2),
        "PI3" => Some(std::f64::consts::FRAC_PI_3),
        "PI4" => Some(std::f64::consts::FRAC_PI_4),
        "PI6" => Some(std::f64::consts::FRAC_PI_6),
        "PI8" => Some(std::f64::consts::FRAC_PI_8),
        "1SQRT2" => Some(std::f64::consts::FRAC_1_SQRT_2),
        "2SQRT_PI" => Some(std::f64::consts::FRAC_2_SQRT_PI),
        "LN2" => Some(std::f64::consts::LN_2),
        "LN10" => Some(std::f64::consts::LN_10),
        _ => None,
    }
}

pub(crate) struct DSPFunctionTranslator<'a, 'b, 'c> {
    dsp_ctx: &'c mut DSPNodeContext,
    dsp_lib: &'b DSPNodeTypeLibrary,
    builder: Option<FunctionBuilder<'a>>,
    variables: HashMap<String, Variable>,
    var_index: usize,
    module: &'a mut JITModule,
    dsp_node_functions: HashMap<String, (Arc<dyn DSPNodeType>, FuncId)>,
    ptr_w: u32,
}

/// Error enum for JIT compilation errors.
#[derive(Debug, Clone)]
pub enum JITCompileError {
    BadDefinedParams,
    UnknownFunction(String),
    UndefinedVariable(String),
    UnknownTable(usize),
    InvalidReturnValueAccess(String),
    DeclareTopFunError(String),
    DefineTopFunError(String),
    UndefinedDSPNode(String),
    UnknownBuffer(usize),
    NoValueBufferWrite(usize),
    NotEnoughArgsInCall(String, u64),
    ModuleError(String),
    NodeStateError(String, u64),
}

macro_rules! b {
    ($self: ident) => {
        $self.builder.as_mut().expect("FunctionBuilder not finalized")
    }
}

impl<'a, 'b, 'c> DSPFunctionTranslator<'a, 'b, 'c> {
    pub fn new(
        dsp_ctx: &'c mut DSPNodeContext,
        dsp_lib: &'b DSPNodeTypeLibrary,
        builder: FunctionBuilder<'a>,
        module: &'a mut JITModule,
    ) -> Self {
        dsp_ctx.init_dsp_function();

        let builder = Some(builder);

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
                    DSPNodeSigBit::DSPStatePtr
                    | DSPNodeSigBit::NodeStatePtr
                    | DSPNodeSigBit::MultReturnPtr => {
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
            b!(self).declare_var(var, typ);
            self.var_index += 1;
        }

        var
    }

    fn translate(&mut self, fun: ASTFun, debug: bool) -> Result<Option<String>, JITCompileError> {
        let ptr_type = self.module.target_config().pointer_type();
        self.ptr_w = ptr_type.bytes();

        let entry_block = b!(self).create_block();
        b!(self).append_block_params_for_function_params(entry_block);
        b!(self).switch_to_block(entry_block);
        b!(self).seal_block(entry_block);

        self.variables.clear();

        // declare and define parameters:
        for param_idx in 0..fun.param_count() {
            let val = b!(self).block_params(entry_block)[param_idx];

            match fun.param_name(param_idx) {
                Some(param_name) => {
                    let var = if fun.param_is_ref(param_idx) {
                        self.declare_variable(ptr_type, param_name)
                    } else {
                        self.declare_variable(F64, param_name)
                    };

                    b!(self).def_var(var, val);
                }
                None => {
                    return Err(JITCompileError::BadDefinedParams);
                }
            }
        }

        // declare and define local variables:
        for local_name in fun.local_variables().iter() {
            let zero = b!(self).ins().f64const(0.0);
            let var = self.declare_variable(F64, local_name);
            b!(self).def_var(var, zero);
        }

        let v = self.compile(fun.ast_ref())?;

        b!(self).ins().return_(&[v]);

        let result = if debug {
            Some(format!("{}", b!(self).func.display()))
        } else {
            None
        };

        self.builder.take().expect("builder not finalized yet").finalize();

        Ok(result)
    }

    fn ins_b_to_f64(&mut self, v: Value) -> Value {
//        let bint = self.b!(self).ins().bint(I32, v);
        b!(self).ins().fcvt_from_uint(F64, v)
    }

    fn compile(&mut self, ast: &ASTNode) -> Result<Value, JITCompileError> {
        match ast {
            ASTNode::Lit(v) => Ok(b!(self).ins().f64const(*v)),
            ASTNode::Var(name) => {
                if let Some(c) = constant_lookup(name) {
                    Ok(b!(self).ins().f64const(c))
                } else if name.starts_with('&') {
                    let variable = self
                        .variables
                        .get(name)
                        .ok_or_else(|| JITCompileError::UndefinedVariable(name.to_string()))?;
                    let ptr = b!(self).use_var(*variable);
                    Ok(b!(self).ins().load(F64, MemFlags::new(), ptr, 0))
                } else if name.starts_with('$') {
                    let aux_vars = self
                        .variables
                        .get("&aux")
                        .ok_or_else(|| JITCompileError::UndefinedVariable("&aux".to_string()))?;

                    let pvs = b!(self).use_var(*aux_vars);
                    let offs = match &name[..] {
                        "$srate" => AUX_VAR_IDX_SRATE,
                        "$israte" => AUX_VAR_IDX_ISRATE,
                        "$reset" => AUX_VAR_IDX_RESET,
                        _ => return Err(JITCompileError::UndefinedVariable(name.to_string())),
                    };
                    let aux_value = b!(self).ins().load(
                        F64,
                        MemFlags::new(),
                        pvs,
                        Offset32::new(offs as i32 * F64.bytes() as i32),
                    );
                    Ok(aux_value)
                } else if name.starts_with('*') {
                    let pv_index = self
                        .dsp_ctx
                        .get_persistent_variable_index(name)
                        .map_err(|_| JITCompileError::UndefinedVariable(name.to_string()))?;

                    let persistent_vars = self
                        .variables
                        .get("&pv")
                        .ok_or_else(|| JITCompileError::UndefinedVariable("&pv".to_string()))?;
                    let pvs = b!(self).use_var(*persistent_vars);
                    let pers_value = b!(self).ins().load(
                        F64,
                        MemFlags::new(),
                        pvs,
                        Offset32::new(pv_index as i32 * F64.bytes() as i32),
                    );
                    Ok(pers_value)
                } else if name.starts_with('%') {
                    if name.len() > 2 {
                        return Err(JITCompileError::InvalidReturnValueAccess(name.to_string()));
                    }

                    let offs: i32 = match name.chars().nth(1) {
                        Some('1') => 0,
                        Some('2') => 1,
                        Some('3') => 2,
                        Some('4') => 3,
                        Some('5') => 4,
                        _ => {
                            return Err(JITCompileError::InvalidReturnValueAccess(
                                name.to_string(),
                            ));
                        }
                    };

                    let return_vals = self
                        .variables
                        .get("&rv")
                        .ok_or_else(|| JITCompileError::UndefinedVariable("&rv".to_string()))?;
                    let rvs = b!(self).use_var(*return_vals);
                    let ret_value = b!(self).ins().load(
                        F64,
                        MemFlags::new(),
                        rvs,
                        Offset32::new(offs * F64.bytes() as i32),
                    );
                    Ok(ret_value)
                } else {
                    let variable = self
                        .variables
                        .get(name)
                        .ok_or_else(|| JITCompileError::UndefinedVariable(name.to_string()))?;
                    Ok(b!(self).use_var(*variable))
                }
            }
            ASTNode::Assign(name, ast) => {
                let value = self.compile(ast)?;

                if name.starts_with('&') {
                    let variable = self
                        .variables
                        .get(name)
                        .ok_or_else(|| JITCompileError::UndefinedVariable(name.to_string()))?;
                    let ptr = b!(self).use_var(*variable);
                    b!(self).ins().store(MemFlags::new(), value, ptr, 0);
                } else if name.starts_with('*') {
                    let pv_index = self
                        .dsp_ctx
                        .get_persistent_variable_index(name)
                        .map_err(|_| JITCompileError::UndefinedVariable(name.to_string()))?;

                    let persistent_vars = self
                        .variables
                        .get("&pv")
                        .ok_or_else(|| JITCompileError::UndefinedVariable("&pv".to_string()))?;
                    let pvs = b!(self).use_var(*persistent_vars);
                    b!(self).ins().store(
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
                    b!(self).def_var(*variable, value);
                }

                Ok(value)
            }
            ASTNode::BinOp(op, a, b) => {
                let value_a = self.compile(a)?;
                let value_b = self.compile(b)?;
                let value = match op {
                    ASTBinOp::Add => b!(self).ins().fadd(value_a, value_b),
                    ASTBinOp::Sub => b!(self).ins().fsub(value_a, value_b),
                    ASTBinOp::Mul => b!(self).ins().fmul(value_a, value_b),
                    ASTBinOp::Div => b!(self).ins().fdiv(value_a, value_b),
                    ASTBinOp::Eq => {
                        let cmp_res = b!(self).ins().fcmp(FloatCC::Equal, value_a, value_b);
                        self.ins_b_to_f64(cmp_res)
                    }
                    ASTBinOp::Ne => {
                        let cmp_res = b!(self).ins().fcmp(FloatCC::Equal, value_a, value_b);
                        let bnot = b!(self).ins().bnot(cmp_res);
//                        let bint = b!(self).ins().bint(I32, bnot);
                        b!(self).ins().fcvt_from_uint(F64, bnot)
                    }
                    ASTBinOp::Ge => {
                        let cmp_res =
                            b!(self).ins().fcmp(FloatCC::GreaterThanOrEqual, value_a, value_b);
                        self.ins_b_to_f64(cmp_res)
                    }
                    ASTBinOp::Le => {
                        let cmp_res =
                            b!(self).ins().fcmp(FloatCC::LessThanOrEqual, value_a, value_b);
                        self.ins_b_to_f64(cmp_res)
                    }
                    ASTBinOp::Gt => {
                        let cmp_res =
                            b!(self).ins().fcmp(FloatCC::GreaterThan, value_a, value_b);
                        self.ins_b_to_f64(cmp_res)
                    }
                    ASTBinOp::Lt => {
                        let cmp_res = b!(self).ins().fcmp(FloatCC::LessThan, value_a, value_b);
                        self.ins_b_to_f64(cmp_res)
                    }
                };

                Ok(value)
            }
            ASTNode::BufDeclare { buf_idx, len } => {
                if *buf_idx >= self.dsp_ctx.config.buffer_count {
                    return Err(JITCompileError::UnknownBuffer(*buf_idx));
                }

                self.dsp_ctx.buffer_declare[*buf_idx] = *len;

                Ok(b!(self).ins().f64const(0.0))
            },
            ASTNode::Len(op) => {
                let (buf_idx, buf_lens) = match op {
                    ASTLenOp::Buffer(buf_idx) => {
                        if *buf_idx >= self.dsp_ctx.config.buffer_count {
                            return Err(JITCompileError::UnknownBuffer(*buf_idx));
                        }

                        let buf_lens = self.variables.get("&buf_lens").ok_or_else(|| {
                            JITCompileError::UndefinedVariable("&buf_lens".to_string())
                        })?;

                        (*buf_idx, buf_lens)
                    },
                    ASTLenOp::Table(tbl_idx) => {
                        let tbl_lens = self.variables.get("&table_lens").ok_or_else(|| {
                            JITCompileError::UndefinedVariable("&table_lens".to_string())
                        })?;

                        if *tbl_idx >= self.dsp_ctx.config.tables.len() {
                            return Err(JITCompileError::UnknownTable(*tbl_idx));
                        }

                        (*tbl_idx, tbl_lens)
                    },
                };

                let lenptr = b!(self).use_var(*buf_lens);
                let len = b!(self).ins().load(
                    I64,
                    MemFlags::new(),
                    lenptr,
                    Offset32::new(buf_idx as i32 * self.ptr_w as i32),
                );

                Ok(b!(self).ins().fcvt_from_uint(F64, len))
            },
            ASTNode::BufOp { op, idx, val } => {
                let idx = self.compile(idx)?;

                let ptr_type = self.module.target_config().pointer_type();

                let (buf_var, buf_idx, buf_lens) = match op {
                    ASTBufOp::Write(buf_idx)
                    | ASTBufOp::Read(buf_idx)
                    | ASTBufOp::ReadLin(buf_idx) => {
                        let buf_var = self.variables.get("&bufs").ok_or_else(|| {
                            JITCompileError::UndefinedVariable("&bufs".to_string())
                        })?;

                        let buf_lens = self.variables.get("&buf_lens").ok_or_else(|| {
                            JITCompileError::UndefinedVariable("&buf_lens".to_string())
                        })?;

                        if *buf_idx >= self.dsp_ctx.config.buffer_count {
                            return Err(JITCompileError::UnknownBuffer(*buf_idx));
                        }

                        (buf_var, buf_idx, buf_lens)
                    }
                    ASTBufOp::TableRead(tbl_idx) | ASTBufOp::TableReadLin(tbl_idx) => {
                        let buf_var = self.variables.get("&tables").ok_or_else(|| {
                            JITCompileError::UndefinedVariable("&tables".to_string())
                        })?;

                        let tbl_lens = self.variables.get("&table_lens").ok_or_else(|| {
                            JITCompileError::UndefinedVariable("&table_lens".to_string())
                        })?;

                        if *tbl_idx >= self.dsp_ctx.config.tables.len() {
                            return Err(JITCompileError::UnknownTable(*tbl_idx));
                        }

                        (buf_var, tbl_idx, tbl_lens)
                    }
                };

                let bptr = b!(self).use_var(*buf_var);
                let buffer = b!(self).ins().load(
                    ptr_type,
                    MemFlags::new(),
                    bptr,
                    Offset32::new(*buf_idx as i32 * self.ptr_w as i32),
                );

                let lenptr = b!(self).use_var(*buf_lens);
                let len = b!(self).ins().load(
                    I64,
                    MemFlags::new(),
                    lenptr,
                    Offset32::new(*buf_idx as i32 * self.ptr_w as i32),
                );

                let orig_idx = idx;
                let idx = b!(self).ins().floor(idx);
                let orig_fint_idx = idx;
                let idx = b!(self).ins().fcvt_to_uint(I64, idx);
                let orig_int_idx = idx;

                let data_width =
                    match op {
                        ASTBufOp::TableReadLin { .. } | ASTBufOp::TableRead { .. } => F32.bytes() as i64,
                        _ => F64.bytes() as i64
                    };

                let idx = b!(self).ins().urem(idx, len);
                let idx = b!(self).ins().imul_imm(idx, data_width);
                let ptr = b!(self).ins().iadd(buffer, idx);

                match op {
                    ASTBufOp::Write { .. } => {
                        let val = val
                            .as_ref()
                            .ok_or_else(|| JITCompileError::NoValueBufferWrite(*buf_idx))?;
                        let val = self.compile(val)?;

                        b!(self).ins().store(MemFlags::new(), val, ptr, 0);
                        Ok(b!(self).ins().f64const(0.0))
                    }
                    ASTBufOp::Read { .. } => {
                        Ok(b!(self).ins().load(F64, MemFlags::new(), ptr, 0))
                    }
                    ASTBufOp::TableRead { .. } => {
                        let sample = b!(self).ins().load(F32, MemFlags::new(), ptr, 0);
                        Ok(b!(self).ins().fpromote(F64, sample))
                    }
                    ASTBufOp::ReadLin { .. } | ASTBufOp::TableReadLin { .. } => {
                        let fract = b!(self).ins().fsub(orig_idx, orig_fint_idx);
                        let idx = b!(self).ins().iadd_imm(orig_int_idx, 1 as i64);
                        let idx = b!(self).ins().urem(idx, len);
                        let idx = b!(self).ins().imul_imm(idx, data_width);
                        let ptr2 = b!(self).ins().iadd(buffer, idx);

                        let (a, b) =
                            if data_width == (I32.bytes() as i64) {
                                let a = b!(self).ins().load(F32, MemFlags::new(), ptr, 0);
                                let b = b!(self).ins().load(F32, MemFlags::new(), ptr2, 0);
                                let a = b!(self).ins().fpromote(F64, a);
                                let b = b!(self).ins().fpromote(F64, b);
                                (a, b)
                            } else {
                                let a = b!(self).ins().load(F64, MemFlags::new(), ptr, 0);
                                let b = b!(self).ins().load(F64, MemFlags::new(), ptr2, 0);
                                (a, b)
                            };
                        let one = b!(self).ins().f64const(1.0);
                        let fract_1 = b!(self).ins().fsub(one, fract);
                        let a = b!(self).ins().fmul(a, fract_1);
                        let b = b!(self).ins().fmul(b, fract);
                        Ok(b!(self).ins().fadd(a, b))
                    }
                }
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
                            dsp_node_fun_params.push(b!(self).use_var(*state_var));
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
                            let fptr = b!(self).use_var(*fstate_var);
                            let func_state = b!(self).ins().load(
                                ptr_type,
                                MemFlags::new(),
                                fptr,
                                Offset32::new(node_state_index as i32 * self.ptr_w as i32),
                            );
                            dsp_node_fun_params.push(func_state);
                        }
                        DSPNodeSigBit::MultReturnPtr => {
                            let ret_var = self.variables.get("&rv").ok_or_else(|| {
                                JITCompileError::UndefinedVariable("&rv".to_string())
                            })?;
                            dsp_node_fun_params.push(b!(self).use_var(*ret_var));
                        }
                    }

                    i += 1;
                }

                let local_callee = self.module.declare_func_in_func(func_id, b!(self).func);
                let call = b!(self).ins().call(local_callee, &dsp_node_fun_params);
                if node_type.has_return_value() {
                    Ok(b!(self).inst_results(call)[0])
                } else {
                    Ok(b!(self).ins().f64const(0.0))
                }
            }
            ASTNode::If(cond, then, els) => {
                let condition_value = if let ASTNode::BinOp(op, a, b) = cond.as_ref() {
                    let val = match op {
                        ASTBinOp::Eq => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            b!(self).ins().fcmp(FloatCC::Equal, a, b)
                        }
                        ASTBinOp::Ne => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            let eq = b!(self).ins().fcmp(FloatCC::Equal, a, b);
                            b!(self).ins().bnot(eq)
                        }
                        ASTBinOp::Gt => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            b!(self).ins().fcmp(FloatCC::GreaterThan, a, b)
                        }
                        ASTBinOp::Lt => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            b!(self).ins().fcmp(FloatCC::LessThan, a, b)
                        }
                        ASTBinOp::Ge => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            b!(self).ins().fcmp(FloatCC::GreaterThanOrEqual, a, b)
                        }
                        ASTBinOp::Le => {
                            let a = self.compile(a)?;
                            let b = self.compile(b)?;
                            b!(self).ins().fcmp(FloatCC::LessThanOrEqual, a, b)
                        }
                        _ => self.compile(cond)?,
                    };

                    val
                } else {
                    let res = self.compile(cond)?;
                    let cmpv = b!(self).ins().f64const(0.5);
                    b!(self).ins().fcmp(FloatCC::GreaterThanOrEqual, res, cmpv)
                };

                let then_block = b!(self).create_block();
                let else_block = b!(self).create_block();
                let merge_block = b!(self).create_block();

                // If-else constructs in the toy language have a return value.
                // In traditional SSA form, this would produce a PHI between
                // the then and else bodies. Cranelift uses block parameters,
                // so set up a parameter in the merge block, and we'll pass
                // the return values to it from the branches.
                b!(self).append_block_param(merge_block, F64);

                // Test the if condition and conditionally branch.
                b!(self).ins().brif(condition_value, then_block, &[], else_block, &[]);

                b!(self).switch_to_block(then_block);
                b!(self).seal_block(then_block);
                let then_return = self.compile(then)?;

                // Jump to the merge block, passing it the block return value.
                b!(self).ins().jump(merge_block, &[then_return]);

                b!(self).switch_to_block(else_block);
                b!(self).seal_block(else_block);
                let else_return = if let Some(els) = els {
                    self.compile(els)?
                } else {
                    b!(self).ins().f64const(0.0)
                };

                // Jump to the merge block, passing it the block return value.
                b!(self).ins().jump(merge_block, &[else_return]);

                // Switch to the merge block for subsequent statements.
                b!(self).switch_to_block(merge_block);

                // We've now seen all the predecessors of the merge block.
                b!(self).seal_block(merge_block);

                // Read the value of the if-else by reading the merge block
                // parameter.
                let phi = b!(self).block_params(merge_block)[0];

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
                    Ok(b!(self).ins().f64const(0.0))
                }
            }
        }
    }
}

/// Returns a [DSPFunction] that does nothing. This can be helpful for initializing
/// structures you want to send to the DSP thread.
pub fn get_nop_function(
    lib: Rc<RefCell<DSPNodeTypeLibrary>>,
    dsp_ctx: Rc<RefCell<DSPNodeContext>>,
) -> Box<DSPFunction> {
    let jit = JIT::new(lib, dsp_ctx);
    jit.compile(ASTFun::new(Box::new(ASTNode::Lit(0.0)))).expect("No compile error")
}
