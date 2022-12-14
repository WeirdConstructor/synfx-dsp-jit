// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

/// Binary operator for the AST
#[derive(Debug, Clone, Copy)]
pub enum ASTBinOp {
    /// Addition "+"
    Add,
    /// Substraction "-"
    Sub,
    /// Multiplication "*"
    Mul,
    /// Division "/"
    Div,
    /// Equals comparison "=="
    Eq,
    /// Not Equals comparison "!="
    Ne,
    /// Less than comparison "<"
    Lt,
    /// Less or equal comparison "<="
    Le,
    /// Greater than comparison ">"
    Gt,
    /// Greater or equal comparison ">="
    Ge,
}

/// Buffer and Table operations
#[derive(Debug, Clone, Copy)]
pub enum ASTBufOp {
    /// Write to sample position, the argument is the buffer index
    Write(usize),
    /// Read from sample position, the argument is the buffer index
    Read(usize),
    /// Read with linear interpolation, the argument is the buffer index
    ReadLin(usize),
    /// Read from a read-only table, the argument is the table index
    TableRead(usize),
    /// Read from an read-only table with linear interpolation, the argument is the table index
    TableReadLin(usize),
}

/// Length of a buffer or table for `ASTNode::Len(...)`
#[derive(Debug, Clone, Copy)]
pub enum ASTLenOp {
    /// Length of a buffer is queries, argument is buffer index
    Buffer(usize),
    /// Length of a table is queries, argument is table index
    Table(usize),
}

/// Top level structure that holds an AST.
///
/// It holds the names of the local variables. For now you can't
/// specify your own parameter names or change the number of
/// parameters. The [crate::DSPFunction] is fixed currently to
/// 8 input parameters with two output signals and one return value (a third signal so to say).
#[derive(Debug, Clone)]
pub struct ASTFun {
    params: Vec<String>,
    locals: Vec<String>,
    ast: Box<ASTNode>,
}

impl ASTFun {
    pub fn new(ast: Box<ASTNode>) -> Self {
        let mut ast_fun = Self {
            params: vec![
                "in1".to_string(),
                "in2".to_string(),
                "alpha".to_string(),
                "beta".to_string(),
                "delta".to_string(),
                "gamma".to_string(),
                "&sig1".to_string(),
                "&sig2".to_string(),
                "&aux".to_string(),
                "&state".to_string(),
                "&fstate".to_string(),
                "&pv".to_string(),
                "&rv".to_string(),
                "&bufs".to_string(),
                "&buf_lens".to_string(),
                "&tables".to_string(),
                "&table_lens".to_string(),
            ],
            locals: vec![], // vec!["x".to_string(), "y".to_string()],
            ast,
        };

        ast_fun.retrieve_local_variable_names();

        ast_fun
    }

    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    pub fn param_is_ref(&self, idx: usize) -> bool {
        if let Some(param_name) = self.params.get(idx) {
            param_name.chars().next() == Some('&')
        } else {
            false
        }
    }

    pub fn param_name(&self, idx: usize) -> Option<&str> {
        self.params.get(idx).map(|s| &s[..])
    }

    pub fn name_is_local_var(&self, name: &str) -> bool {
        for param in self.params.iter() {
            if name == param {
                return false;
            }
            if name.chars().next() == Some('*') {
                return false;
            }
        }

        true
    }

    pub fn local_variables(&self) -> &Vec<String> {
        &self.locals
    }

    pub fn retrieve_local_variable_names(&mut self) -> &Vec<String> {
        self.locals.clear();

        let mut ast = Box::new(ASTNode::Lit(1.0));
        std::mem::swap(&mut ast, &mut self.ast);

        walk_ast(ast.as_mut(), &mut |node| {
            if let ASTNode::Var(name) = node {
                if self.name_is_local_var(&name) {
                    if !self.locals.contains(name) {
                        self.locals.push(name.to_string());
                    }
                }
            } else if let ASTNode::Assign(name, _) = node {
                if self.name_is_local_var(&name) {
                    if !self.locals.contains(name) {
                        self.locals.push(name.to_string());
                    }
                }
            }
        });

        std::mem::swap(&mut ast, &mut self.ast);

        &self.locals
    }

    pub fn ast_ref(&self) -> &ASTNode {
        &self.ast
    }
}

pub fn walk_ast<F: FnMut(&mut ASTNode)>(node: &mut ASTNode, f: &mut F) {
    f(node);
    match node {
        ASTNode::Lit(_) | ASTNode::Var(_) | ASTNode::BufDeclare { .. } | ASTNode::Len(_) => {}
        ASTNode::Assign(_, expr) => {
            walk_ast(expr.as_mut(), f);
        }
        ASTNode::BinOp(_, expr1, expr2) => {
            walk_ast(expr1.as_mut(), f);
            walk_ast(expr2.as_mut(), f);
        }
        ASTNode::If(expr1, expr2, expr3) => {
            walk_ast(expr1.as_mut(), f);
            walk_ast(expr2.as_mut(), f);
            if let Some(expr3) = expr3 {
                walk_ast(expr3.as_mut(), f);
            }
        }
        ASTNode::Call(_, _, exprs) => {
            for e in exprs.iter_mut() {
                walk_ast(e.as_mut(), f);
            }
        }
        ASTNode::Stmts(stmts) => {
            for s in stmts.iter_mut() {
                walk_ast(s.as_mut(), f);
            }
        }
        ASTNode::BufOp { idx, val, .. } => {
            walk_ast(idx.as_mut(), f);
            if let Some(val) = val {
                walk_ast(val.as_mut(), f);
            }
        }
    }
}

/// The abstract syntax tree that the [crate::JIT] can compile down to machine
/// code (in form of a [crate::DSPFunction]) for you.
///
/// See also the [crate::build] module about creating these trees conveniently
/// directly from Rust code.
#[derive(Debug, Clone)]
pub enum ASTNode {
    /// Literal fixed f64 values.
    Lit(f64),
    /// Variable and parameter names. Variables that start with a "*" are
    /// stored persistently across multiple [crate::DSPFunction] invocations for you.
    Var(String),
    /// Assigns a value to a variable.
    Assign(String, Box<ASTNode>),
    /// A binary operator. See also [ASTBinOp] which operations are possible.
    BinOp(ASTBinOp, Box<ASTNode>, Box<ASTNode>),
    /// A conditional statement.
    ///
    /// You can specify a [ASTBinOp] with a comparison operation as first element
    /// here (the condition), or any other kind of expression that returns a value.
    /// In the latter case the value must be larger or equal to `0.5` to be true.
    If(Box<ASTNode>, Box<ASTNode>, Option<Box<ASTNode>>),
    /// Calls a DSP node/function by it's name. The second parameter here, the `u64`
    /// is the unique ID for this ASTNode. It's used to track state of this DSP node.
    /// You have to make sure that the IDs don't change and that you are not using
    /// the same ID for multiple stateful DSP nodes here.
    Call(String, u64, Vec<Box<ASTNode>>),
    /// Declare the length of a buffer. By default all buffers are only 16 samples long.
    /// If you change the length of a buffer, a new buffer will be allocated on the fly
    /// and sent to the backend. [crate::DSPNodeContext] and [crate::DSPFunction] take care of
    /// disposing the old buffer allocation and preserving the data as good as possible.
    BufDeclare { buf_idx: usize, len: usize },
    /// Perform a buffer or table operation on the specified buffer/table at the given
    /// index with an optional value. Tables and buffers don't share their index space,
    /// each have their own index space. The buffer or table index is passed as the
    /// [ASTBufOp] enumeration.
    BufOp { op: ASTBufOp, idx: Box<ASTNode>, val: Option<Box<ASTNode>> },
    /// To retrieve the length of a buffer or table.
    Len(ASTLenOp),
    /// A list of statements that must be executed in the here specified order.
    Stmts(Vec<Box<ASTNode>>),
}

impl ASTNode {
    pub fn to_string(&self) -> String {
        match self {
            ASTNode::Lit(v) => format!("lit:{:6.4}", v),
            ASTNode::Var(v) => format!("var:{}", v),
            ASTNode::Assign(v, _) => format!("assign:{}", v),
            ASTNode::BinOp(op, _, _) => format!("binop:{:?}", op),
            ASTNode::If(_, _, _) => format!("if"),
            ASTNode::Call(fun, fs, _) => format!("call{}:{}", fs, fun),
            ASTNode::BufDeclare { buf_idx, len } => format!("bufdecl{}:{}", buf_idx, len),
            ASTNode::Len(op) => format!("len:{:?}", op),
            ASTNode::BufOp { op, .. } => format!("buf:{:?}", op),
            ASTNode::Stmts(stmts) => format!("stmts:{}", stmts.len()),
        }
    }

    pub fn typ_str(&self) -> &str {
        match self {
            ASTNode::Lit(_v) => "lit",
            ASTNode::Var(_v) => "var",
            ASTNode::Assign(_v, _) => "assign",
            ASTNode::BinOp(_op, _, _) => "binop",
            ASTNode::If(_, _, _) => "if",
            ASTNode::Call(_fun, _, _) => "call",
            ASTNode::BufDeclare { .. } => "bufdecl",
            ASTNode::Len(_) => "len",
            ASTNode::BufOp { .. } => "bufop",
            ASTNode::Stmts(_stmts) => "stmts",
        }
    }

    /// Tree dump of the AST. Returns a neatly indented tree. Pass 0 as first `indent`.
    pub fn dump(&self, indent: usize) -> String {
        let indent_str = "   ".repeat(indent + 1);
        let mut s = indent_str + &self.to_string() + "\n";

        match self {
            ASTNode::Lit(_) => (),
            ASTNode::Var(_) => (),
            ASTNode::BufDeclare { .. } => (),
            ASTNode::Len(_) => (),
            ASTNode::Assign(_, e) => {
                s += &e.dump(indent + 1);
            }
            ASTNode::BinOp(_, a, b) => {
                s += &a.dump(indent + 1);
                s += &b.dump(indent + 1);
            }
            ASTNode::If(c, a, b) => {
                s += &c.dump(indent + 1);
                s += &a.dump(indent + 1);
                if let Some(n) = b {
                    s += &n.dump(indent + 1);
                }
            }
            ASTNode::Call(_, _, args) => {
                for (i, a) in args.iter().enumerate() {
                    s += &format!("[{}] {}", i, &a.dump(indent + 1));
                }
            }
            ASTNode::Stmts(stmts) => {
                for n in stmts {
                    s += &n.dump(indent + 1);
                }
            }
            ASTNode::BufOp { idx, val, .. } => {
                s += &idx.dump(indent + 1);
                if let Some(val) = val {
                    s += &val.dump(indent + 1);
                }
            }
        }

        s
    }
}

pub mod build {
    /*! This module provides a simplified API for building an [ASTNode] tree
    directly in Rust. It's useful for building inline DSP functions in Rust
    either for your own projects or test cases.
    */
    use super::*;

    pub fn fun(e: Box<ASTNode>) -> ASTFun {
        ASTFun::new(e)
    }

    pub fn literal(v: f64) -> Box<ASTNode> {
        Box::new(ASTNode::Lit(v))
    }

    pub fn var(name: &str) -> Box<ASTNode> {
        Box::new(ASTNode::Var(name.to_string()))
    }

    pub fn assign(name: &str, e: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::Assign(name.to_string(), e))
    }

    pub fn op_eq(a: Box<ASTNode>, b: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BinOp(ASTBinOp::Eq, a, b))
    }

    pub fn op_ne(a: Box<ASTNode>, b: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BinOp(ASTBinOp::Ne, a, b))
    }

    pub fn op_le(a: Box<ASTNode>, b: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BinOp(ASTBinOp::Le, a, b))
    }

    pub fn op_lt(a: Box<ASTNode>, b: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BinOp(ASTBinOp::Lt, a, b))
    }

    pub fn op_ge(a: Box<ASTNode>, b: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BinOp(ASTBinOp::Ge, a, b))
    }

    pub fn op_gt(a: Box<ASTNode>, b: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BinOp(ASTBinOp::Gt, a, b))
    }

    pub fn op_add(a: Box<ASTNode>, b: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BinOp(ASTBinOp::Add, a, b))
    }

    pub fn op_sub(a: Box<ASTNode>, b: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BinOp(ASTBinOp::Sub, a, b))
    }

    pub fn op_mul(a: Box<ASTNode>, b: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BinOp(ASTBinOp::Mul, a, b))
    }

    pub fn op_div(a: Box<ASTNode>, b: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BinOp(ASTBinOp::Div, a, b))
    }

    pub fn buf_declare(buf_idx: usize, len: usize) -> Box<ASTNode> {
        Box::new(ASTNode::BufDeclare { buf_idx, len })
    }

    pub fn buf_len(buf_idx: usize) -> Box<ASTNode> {
        Box::new(ASTNode::Len(ASTLenOp::Buffer(buf_idx)))
    }

    pub fn table_len(tbl_idx: usize) -> Box<ASTNode> {
        Box::new(ASTNode::Len(ASTLenOp::Table(tbl_idx)))
    }

    pub fn buf_write(buf_idx: usize, idx: Box<ASTNode>, val: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BufOp { op: ASTBufOp::Write(buf_idx), idx, val: Some(val) })
    }

    pub fn buf_read(buf_idx: usize, idx: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BufOp { op: ASTBufOp::Read(buf_idx), idx, val: None })
    }

    pub fn buf_read_lin(buf_idx: usize, idx: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BufOp { op: ASTBufOp::ReadLin(buf_idx), idx, val: None })
    }

    pub fn table_read(tbl_idx: usize, idx: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BufOp { op: ASTBufOp::TableRead(tbl_idx), idx, val: None })
    }

    pub fn table_read_lin(tbl_idx: usize, idx: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::BufOp { op: ASTBufOp::TableReadLin(tbl_idx), idx, val: None })
    }

    pub fn stmts(s: &[Box<ASTNode>]) -> Box<ASTNode> {
        Box::new(ASTNode::Stmts(s.to_vec()))
    }

    pub fn call(name: &str, uid: u64, args: &[Box<ASTNode>]) -> Box<ASTNode> {
        Box::new(ASTNode::Call(name.to_string(), uid, args.to_vec()))
    }

    pub fn _if(cond: Box<ASTNode>, a: Box<ASTNode>, b: Option<Box<ASTNode>>) -> Box<ASTNode> {
        Box::new(ASTNode::If(cond, a, b))
    }
}
