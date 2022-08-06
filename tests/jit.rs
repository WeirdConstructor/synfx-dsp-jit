// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

use std::cell::RefCell;
use std::rc::Rc;
use synfx_dsp_jit::*;

mod test_node_type;
use synfx_dsp_jit::DSPState;
use test_node_type::TSTState;

fn get_default_library() -> Rc<RefCell<DSPNodeTypeLibrary>> {
    let lib = get_standard_library();
    lib.borrow_mut().add(std::sync::Arc::new(test_node_type::TestNodeType::default()));
    lib
}

#[macro_export]
macro_rules! assert_float_eq {
    ($a:expr, $b:expr) => {
        if ($a - $b).abs() > 0.0001 {
            panic!(
                r#"assertion failed: `(left == right)`
  left: `{:?}`,
 right: `{:?}`"#,
                $a, $b
            )
        }
    };
}

#[test]
fn check_jit() {
    let dsp_ctx = DSPNodeContext::new_ref();
    let jit = JIT::new(get_default_library(), dsp_ctx.clone());

    let ast = ASTNode::Assign(
        "&sig1".to_string(),
        Box::new(ASTNode::BinOp(
            ASTBinOp::Add,
            Box::new(ASTNode::If(
                Box::new(ASTNode::Var("in2".to_string())),
                Box::new(ASTNode::Call("test".to_string(), 1, vec![Box::new(ASTNode::Lit(11.0))])),
                Some(Box::new(ASTNode::Lit(99.12))),
            )),
            Box::new(ASTNode::Var("in1".to_string())),
        )),
    );

    let fun = ASTFun::new(Box::new(ast));
    let mut code = jit.compile(fun).unwrap();

    code.init(44100.0, None);

    unsafe {
        code.with_dsp_state(|state| {
            (*state).x = 11.0;
            (*state).y = 1.0;
        });
        code.with_node_state(1, |state: *mut TSTState| {
            (*state).l = 44.53;
        })
        .expect("node state exists");
    };
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let res = code.exec(1.0, 0.0, 3.0, 4.0, 5.0, 6.0, &mut s1, &mut s2);
    assert_float_eq!(res, 100.12);
    unsafe {
        code.with_dsp_state(|state| {
            assert_float_eq!((*state).x, 11.0);
        });
    }

    let res2 = code.exec(22.0, 1.0, 3.0, 4.0, 5.0, 6.0, &mut s1, &mut s2);
    assert_float_eq!(res2, 11.0 * 10000.0 + 1.0 + 22.0);
    unsafe {
        code.with_dsp_state(|state| {
            assert_float_eq!((*state).x, 11.0 * 22.0);
            assert_float_eq!((*state).y, 44.53);
        });
    }

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_jit_stmts() {
    let dsp_ctx = DSPNodeContext::new_ref();
    let jit = JIT::new(get_default_library(), dsp_ctx.clone());

    use synfx_dsp_jit::build::*;

    let fun = fun(stmts(&[assign("&sig1", var("in2")), assign("&sig2", var("in1"))]));

    let mut code = jit.compile(fun).unwrap();
    code.init(44100.0, None);

    let (s1, s2, res) = code.exec_2in_2out(1.1, 2.2);
    assert_float_eq!(res, 1.1);
    assert_float_eq!(s1, 2.2);
    assert_float_eq!(s2, 1.1);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_jit_thread_stmts() {
    let dsp_ctx = DSPNodeContext::new_ref();
    let jit = JIT::new(get_default_library(), dsp_ctx.clone());
    use synfx_dsp_jit::build::*;

    let fun = fun(stmts(&[
        assign("&sig1", call("sin", 0, &[var("in2")])),
        assign("&sig2", op_add(literal(23.0), var("in1"))),
    ]));

    let (tx, rx) = std::sync::mpsc::channel();

    let mut code = jit.compile(fun).unwrap();

    std::thread::spawn(move || {
        code.init(44100.0, None);
        let (s1, s2, res) = code.exec_2in_2out(1.1, 2.2);
        tx.send((s1, s2, res)).expect("Sending via mpsc works here");
    })
    .join()
    .expect("Joining threads works in this test");

    let (s1, s2, res) = rx.recv().unwrap();
    assert_float_eq!(res, 24.1);
    assert_float_eq!(s1, 0.80849);
    assert_float_eq!(s2, 24.1);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_jit_sin_2() {
    let ctx = DSPNodeContext::new_ref();
    let jit = JIT::new(get_default_library(), ctx.clone());

    let ast = ASTNode::Call("sin".to_string(), 0, vec![Box::new(ASTNode::Lit(0.5 * 3.14))]);
    let fun = ASTFun::new(Box::new(ast));
    let mut code = jit.compile(fun).unwrap();
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let res = code.exec(1.1, 2.2, 3.0, 4.0, 5.0, 6.0, &mut s1, &mut s2);
    assert_float_eq!(res, 1.0);

    ctx.borrow_mut().free();
}

#[test]
fn check_jit_build_ast() {
    use synfx_dsp_jit::build::*;
    let ast = _if(op_gt(var("in1"), literal(10.0)), literal(1.2), Some(var("in2")));

    let ctx = DSPNodeContext::new_ref();
    let jit = JIT::new(get_default_library(), ctx.clone());

    let mut code = jit.compile(ASTFun::new(ast)).unwrap();

    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let ret = code.exec(20.21, 3.4, 3.0, 4.0, 5.0, 6.0, &mut s1, &mut s2);
    assert_float_eq!(ret, 1.2);

    let ret = code.exec(2.21, 3.4, 3.0, 4.0, 5.0, 6.0, &mut s1, &mut s2);
    assert_float_eq!(ret, 3.4);

    ctx.borrow_mut().free();
}

#[test]
fn check_jit_sin_1() {
    use synfx_dsp_jit::build::*;
    let ast = call("sin", 0, &[var("in1")]);

    let ctx = DSPNodeContext::new_ref();
    let jit = JIT::new(get_default_library(), ctx.clone());

    let mut code = jit.compile(ASTFun::new(ast)).unwrap();

    let mut s1 = 0.0;
    let mut s2 = 0.0;
    use std::time::Instant;

    let now = Instant::now();
    let mut sum1 = 0.0;
    for _i in 0..10000000 {
        let ret = code.exec(2.21, 3.4, 3.0, 4.0, 5.0, 6.0, &mut s1, &mut s2);
        sum1 += ret;
    }
    println!("SUM JIT: {} time: {}", sum1, now.elapsed().as_millis());

    let now = Instant::now();
    let mut sum2 = 0.0;
    for _i in 0..10000000 {
        let ret = (2.21_f64).sin();
        sum2 += ret;
    }
    println!("SUM RST: {} time: {}", sum2, now.elapsed().as_millis());
    ctx.borrow_mut().free();

    assert_float_eq!(sum1, sum2);
}

fn exec_ast(ast: Box<ASTNode>, in1: f64, in2: f64) -> (f64, f64, f64) {
    let dsp_ctx = DSPNodeContext::new_ref();
    dsp_ctx.borrow_mut().set_debug(true);
    let jit = JIT::new(get_default_library(), dsp_ctx.clone());

    let mut code = jit.compile(ASTFun::new(ast)).unwrap();
    println!("DEBUG: {}", dsp_ctx.borrow().get_ir_dump());

    code.init(44100.0, None);
    let ret = code.exec_2in_2out(in1, in2);

    dsp_ctx.borrow_mut().free();
    ret
}

#[test]
fn check_jit_sample_rate_vars() {
    use synfx_dsp_jit::build::*;
    let (_, _, ret) = exec_ast(var("$srate"), 0.0, 0.0);
    assert_float_eq!(ret, 44100.0);
    let (_, _, ret) = exec_ast(var("$israte"), 0.0, 0.0);
    assert_float_eq!(ret, 1.0 / 44100.0);
}

struct MyDSPNode {
    value: f64,
}

impl MyDSPNode {
    fn reset(&mut self, _state: &mut DSPState) {
        *self = Self::default();
    }
}

impl Default for MyDSPNode {
    fn default() -> Self {
        Self { value: 0.0 }
    }
}

extern "C" fn process_my_dsp_node(my_state: *mut MyDSPNode) -> f64 {
    let mut my_state = unsafe { &mut *my_state };
    my_state.value += 1.0;
    my_state.value
}

synfx_dsp_jit::stateful_dsp_node_type! {
    DIYNodeType, MyDSPNode => process_my_dsp_node "my_dsp" "Sr"
    doc "Simple counter node. Counts in increments of 1.0 each time it's called."
    inputs
    outputs 0 "sum"
}

#[test]
fn check_self_defined_nodes() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    let lib = get_default_library();

    lib.borrow_mut().add(DIYNodeType::new_ref());

    // Compile first version of the DSP code, just one instance of the
    // "my_dsp" node:
    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit.compile(ASTFun::new(call("my_dsp", 0, &[]))).unwrap();

    code.init(44100.0, None);

    let (_, _, ret) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(ret, 1.0); // my_dsp counter returned 1.0

    // Compile second version of the DSP code, now with two instances:
    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let ast2 =
        stmts(&[assign("&sig1", call("my_dsp", 0, &[])), assign("&sig2", call("my_dsp", 1, &[]))]);
    let mut code = jit.compile(ASTFun::new(ast2)).unwrap();

    code.init(44100.0, None);
    let (s1, s2, _) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 2.0); // previous instance of my_dsp returns 2.0
    assert_float_eq!(s2, 1.0); // new instance returns 1.0

    let (s1, s2, _) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 3.0); // previous instance of my_dsp returns 3.0
    assert_float_eq!(s2, 2.0); // new instance returns 2.0

    // Now we reset the state:
    code.reset();
    let (s1, s2, _) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 1.0); // Now both have the same starting value
    assert_float_eq!(s2, 1.0); // Now both have the same starting value

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_persistent_vars() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    let lib = get_default_library();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[
            assign("*a", literal(10.1)),
            assign("*b", literal(12.03)),
            var("*a"),
        ])))
        .unwrap();

    code.init(44100.0, None);

    let (_, _, ret) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(ret, 10.1);

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code2 = jit
        .compile(ASTFun::new(stmts(&[assign("*c", op_add(var("*a"), var("*b"))), var("*c")])))
        .unwrap();
    code2.init(44100.0, Some(&code));

    let (_, _, ret) = code2.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(ret, 22.13);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_multi_returns() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    let lib = get_default_library();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[
            assign("&sig1", call("test", 1, &[literal(1.0)])),
            assign("&sig2", var("%3")),
            op_add(op_add(var("%1"), var("%2")), op_add(var("%4"), var("%5"))),
        ])))
        .unwrap();

    code.init(44100.0, None);

    let (s1, s2, ret) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 10001.0);
    assert_float_eq!(s2, 92.2);
    assert_float_eq!(ret, 90.4 + 91.3 + 93.1 + 94.0);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_phasor_example() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    let lib = get_default_library();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[
            assign("*phase", op_add(var("*phase"), op_mul(literal(440.0), var("$israte")))),
            _if(
                op_gt(var("*phase"), literal(1.0)),
                assign("*phase", op_sub(var("*phase"), literal(1.0))),
                None,
            ),
            var("*phase"),
        ])))
        .unwrap();

    code.init(44100.0, None);

    let mut out = vec![];
    for i in 0..200 {
        let (_, _, ret) = code.exec_2in_2out(0.0, 0.0);
        if i % 20 == 0 {
            out.push(ret);
        }
    }
    assert_float_eq!(out[0], 0.0099);
    assert_float_eq!(out[1], 0.2095);
    assert_float_eq!(out[2], 0.4090);
    assert_float_eq!(out[3], 0.6086);
    assert_float_eq!(out[4], 0.8081);
    assert_float_eq!(out[5], 0.0077);
    assert_float_eq!(out[6], 0.2072);
    assert_float_eq!(out[7], 0.4068);
    assert_float_eq!(out[8], 0.6063);
    assert_float_eq!(out[9], 0.8058);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_documentation_exists() {
    let lib = get_default_library();
    let nt = lib.borrow().get_type_by_name("accum").expect("Type 'accum' exists");
    assert!(nt.documentation().find("reset").is_some());
    assert_eq!(nt.input_names(0), Some("input"));
    assert_eq!(nt.input_names(1), Some("reset"));
    assert_eq!(nt.input_names(2), None);
    assert_eq!(nt.input_index_by_name("reset"), Some(1));
    assert_eq!(nt.output_names(0), Some("sum"));
    assert_eq!(nt.output_names(1), None);
}

#[test]
fn check_node_accum() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    let lib = get_default_library();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[assign(
            "&sig1",
            call("accum", 1, &[literal(0.1), literal(0.0)]),
        )])))
        .unwrap();

    code.init(44100.0, None);

    let (_s1, _s2, _ret) = code.exec_2in_2out(0.0, 0.0);
    let (_s1, _s2, _ret) = code.exec_2in_2out(0.0, 0.0);
    let (s1, _s2, _ret) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 0.3);

    let old_code = code;
    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[assign(
            "&sig1",
            call("accum", 1, &[literal(0.1), literal(1.0)]),
        )])))
        .unwrap();

    code.init(44100.0, Some(&old_code));
    let (s1, _s2, _ret) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 0.0);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_persistent_var_access() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    let lib = get_default_library();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[
            assign("*test", literal(11.0)),
            op_add(var("*ofof"), var("*iiii")),
        ])))
        .unwrap();

    code.init(44100.0, None);

    let (_, _, ret) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(ret, 0.0);

    let index_test = dsp_ctx.borrow().get_persistent_variable_index_by_name("*test").unwrap();
    let test_value = code.access_persistent_var(index_test).map(|var| *var).unwrap();
    assert_float_eq!(test_value, 11.0);

    let index_iiii = dsp_ctx.borrow().get_persistent_variable_index_by_name("*iiii").unwrap();
    let index_ofof = dsp_ctx.borrow().get_persistent_variable_index_by_name("*ofof").unwrap();
    code.access_persistent_var(index_iiii).map(|var| *var = 99.0);

    let (_, _, ret) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(ret, 99.0);

    code.access_persistent_var(index_ofof).map(|var| *var = 1.5);

    let (_, _, ret) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(ret, 100.5);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_stdlib_divrem() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    let lib = get_default_library();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[
            assign("&sig1", call("/%", 1, &[literal(3.4), literal(2.0)])),
            assign("&sig2", var("%1")),
        ])))
        .unwrap();

    code.init(44100.0, None);

    let (s1, s2, _ret) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 1.7);
    assert_float_eq!(s2, 1.4);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_stdlib_phase() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    let lib = get_default_library();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[assign("&sig1", call("phase", 1, &[literal(1000.0)]))])))
        .unwrap();

    code.init(44100.0, None);

    for _ in 0..11 {
        code.exec_2in_2out(0.0, 0.0);
    }
    let (s1, _, _) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 0.2721);

    for _ in 0..10 {
        code.exec_2in_2out(0.0, 0.0);
    }
    let (s1, _, _) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 0.5215);

    for _ in 0..8 {
        code.exec_2in_2out(0.0, 0.0);
    }
    let (s1, _, _) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 0.7256);

    code.reset();

    for _ in 0..11 {
        code.exec_2in_2out(0.0, 0.0);
    }
    let (s1, _, _) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 0.2721);

    for _ in 0..110 {
        code.exec_2in_2out(0.0, 0.0);
    }
    let (s1, _, _) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 0.7891);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_stdlib_atomr() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    let lib = get_default_library();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[
            assign("&sig1", call("atomr", 0, &[literal(0.0)])),
            assign("&sig2", call("atomr", 0, &[literal(1.0)])),
            call("atomr", 0, &[literal(0.5)]),
        ])))
        .unwrap();

    code.init(44100.0, None);

    dsp_ctx.borrow().atom(0).unwrap().set(12.0);
    dsp_ctx.borrow().atom(1).unwrap().set(14.0);

    let (s1, s2, ret) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 12.0);
    assert_float_eq!(ret, 12.0);
    assert_float_eq!(s2, 14.0);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_stdlib_atomr_lin() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    let lib = get_default_library();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[
            assign("&sig1", call("atomr~", 0, &[literal(0.0)])),
            assign("&sig2", call("atomr~", 0, &[literal(1.0)])),
            call("atomr~", 0, &[literal(0.5)]),
        ])))
        .unwrap();

    code.init(44100.0, None);

    dsp_ctx.borrow().atom(0).unwrap().set(12.0);
    dsp_ctx.borrow().atom(1).unwrap().set(14.0);

    let (s1, s2, ret) = code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(s1, 12.0);
    assert_float_eq!(ret, 13.0);
    assert_float_eq!(s2, 14.0);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_stdlib_atomw() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    let lib = get_default_library();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[
            assign("&sig1", call("atomw", 0, &[literal(0.0), var("in1")])),
            assign("&sig2", call("atomw", 0, &[literal(1.0), var("in2")])),
        ])))
        .unwrap();

    code.init(44100.0, None);

    let (s1, s2, _) = code.exec_2in_2out(12.0, 13.0);
    assert_float_eq!(s1, 12.0);
    assert_float_eq!(s2, 13.0);
    let x1 = dsp_ctx.borrow().atom(0).unwrap().get();
    let x2 = dsp_ctx.borrow().atom(1).unwrap().get();
    assert_float_eq!(x1, 12.0);
    assert_float_eq!(x2, 13.0);

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_constants() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    dsp_ctx.borrow_mut().set_debug(true);
    let lib = get_default_library();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[
            assign("*1", var("PI")),
            assign("*2", var("TAU")),
            assign("*3", var("E")),
            assign("*4", var("1PI")),
            assign("*5", var("2PI")),
            assign("*6", var("PI2")),
            assign("*7", var("PI3")),
            assign("*8", var("PI4")),
            assign("*9", var("PI6")),
            assign("*10", var("PI8")),
            assign("*11", var("1SQRT2")),
            assign("*12", var("2SQRT_PI")),
            assign("*13", var("LN2")),
            assign("*14", var("LN10")),
        ])))
        .unwrap();
    println!("DEBUG: {}", dsp_ctx.borrow().get_ir_dump());

    code.init(44100.0, None);
    code.exec_2in_2out(0.0, 0.0);

    let consts = [
        std::f64::consts::PI,
        std::f64::consts::TAU,
        std::f64::consts::E,
        std::f64::consts::FRAC_1_PI,
        std::f64::consts::FRAC_2_PI,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::FRAC_PI_3,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_6,
        std::f64::consts::FRAC_PI_8,
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_2_SQRT_PI,
        std::f64::consts::LN_2,
        std::f64::consts::LN_10,
    ];
    for (i, c) in consts.iter().enumerate() {
        let r = code.access_persistent_var(i).map(|var| *var).unwrap();
        assert_float_eq!(r, c);
    }

    dsp_ctx.borrow_mut().free();
}

#[test]
fn check_reset() {
    use synfx_dsp_jit::build::*;

    let dsp_ctx = DSPNodeContext::new_ref();
    dsp_ctx.borrow_mut().set_debug(true);
    let lib = get_default_library();

    let atom1 = dsp_ctx.borrow().atom(0).unwrap();
    let atom2 = dsp_ctx.borrow().atom(1).unwrap();

    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[
            assign("prev", call("atomr", 0, &[literal(0.0)])),
            call("atomw", 0, &[literal(0.0), op_add(var("prev"), var("$reset"))]),
            call(
                "atomw",
                0,
                &[literal(1.0), op_add(call("atomr", 0, &[literal(1.0)]), literal(1.0))],
            ),
        ])))
        .unwrap();
    println!("DEBUG: {}", dsp_ctx.borrow().get_ir_dump());

    code.init(44100.0, None);
    for _ in 0..10 {
        code.exec_2in_2out(0.0, 0.0);
    }

    // Init does not count as reset, except after the very first one:
    assert_float_eq!(atom1.get(), 1.0);

    code.reset();
    code.exec_2in_2out(0.0, 0.0);
    code.exec_2in_2out(0.0, 0.0);
    code.exec_2in_2out(0.0, 0.0);
    code.exec_2in_2out(0.0, 0.0);

    code.reset();
    code.exec_2in_2out(0.0, 0.0);
    code.exec_2in_2out(0.0, 0.0);
    code.exec_2in_2out(0.0, 0.0);

    assert_float_eq!(atom1.get(), 3.0);
    assert_float_eq!(atom2.get(), 17.0);

    // Recompile to check that we don't get an reset:
    let old_code = code;
    let jit = JIT::new(lib.clone(), dsp_ctx.clone());
    let mut code = jit
        .compile(ASTFun::new(stmts(&[
            assign("prev", call("atomr", 0, &[literal(0.0)])),
            call("atomw", 0, &[literal(0.0), op_add(var("prev"), var("$reset"))]),
            call(
                "atomw",
                0,
                &[literal(1.0), op_add(call("atomr", 0, &[literal(1.0)]), literal(1.0))],
            ),
        ])))
        .unwrap();

    code.init(44100.0, Some(&old_code));

    code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(atom1.get(), 3.0);

    code.reset();
    code.exec_2in_2out(0.0, 0.0);
    assert_float_eq!(atom1.get(), 4.0);

    dsp_ctx.borrow_mut().free();
}
