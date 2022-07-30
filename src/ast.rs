// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

#[derive(Debug, Clone, Copy)]
pub enum ASTBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

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
                "srate".to_string(),
                "israte".to_string(),
                "&sig1".to_string(),
                "&sig2".to_string(),
                "&state".to_string(),
                "&fstate".to_string(),
                "&pv".to_string(),
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
                    self.locals.push(name.to_string());
                }
            }
        });

        std::mem::swap(&mut ast, &mut self.ast);

        &self.locals
    }

    pub fn ast_ref(&self) -> &Box<ASTNode> {
        &self.ast
    }
}

fn walk_ast<F: FnMut(&mut ASTNode)>(node: &mut ASTNode, f: &mut F) {
    f(node);
    match node {
        ASTNode::Lit(_) | ASTNode::Var(_) => {}
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
    }
}

#[derive(Debug, Clone)]
pub enum ASTNode {
    Lit(f64),
    Var(String),
    Assign(String, Box<ASTNode>),
    BinOp(ASTBinOp, Box<ASTNode>, Box<ASTNode>),
    If(Box<ASTNode>, Box<ASTNode>, Option<Box<ASTNode>>),
    Call(String, u64, Vec<Box<ASTNode>>),
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
            ASTNode::Stmts(_stmts) => "stmts",
        }
    }

    pub fn dump(&self, indent: usize) -> String {
        let indent_str = "   ".repeat(indent + 1);
        let mut s = indent_str + &self.to_string() + "\n";

        match self {
            ASTNode::Lit(_) => (),
            ASTNode::Var(_) => (),
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
        }

        s
    }
}

pub mod build {
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
