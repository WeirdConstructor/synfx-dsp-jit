# synfx-dsp-jit

synfx-dsp-jit is a specialized JIT compiler for digital (audio) signal processing for Rust.

This library allows you to compile an simplified abstract syntax tree (AST) down to machine code.
This crate uses the [Cranelift JIT compiler](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift)
for this task. For called Rust functions from the JIT code, either in form
of stateful DSP nodes or stateless DSP functions, It removes any dynamic dispatch overhead.

The result is packaged conveniently for you in a [DSPFunction] structure.

One primary feature that is covered by this library is the state management of stateful
nodes/components that can be called from the AST. By attaching a unique ID to your
AST nodes that call stateful components (aka nodes), this library tracks already initialized
nodes. It does this to allow you to re-compile the [DSPFunction] and make changes without
the audio being interrupted (unless your changes interrupt it).

Aside from the compiling process and state management this library also offers
a (growing) standard library of common DSP algorithms.

All this means this library is primarily directed towards the use case within a real time
synthesis environment.

You can practically build your own JIT compiled [Pure Data](https://puredata.info/) or
[SuperCollider](https://supercollider.github.io/) clone with this. In case you
put in the work of implementing all the DSP nodes and put a compiler on top of
this JIT of course. Other notable projects in this direction are:

- BitWig's "The Grid", which seems to use LLVM under the hood, either to AOT compiler the devices
or even JIT compiling the Grid itself (I'm not sure about that).
- [Gammou - polyphonic modular sound synthesizer](https://github.com/aliefhooghe/Gammou)

This library is used for instance by [HexoDSP](https://github.com/WeirdConstructor/HexoDSP),
which is a comprehensive DSP graph and synthesis library for developing a modular
synthesizer in Rust, such as [HexoSynth](https://github.com/WeirdConstructor/HexoSynth).
It is not the core of HexoDSP, but only provides a small optional part though.

### Quick Start API

To get you started quickly and learn how to use the API I recommend the [instant_compile_ast]
function. But be aware that this function recreates the whole [DSPNodeContext]
on each compilation, so there is no state tracking for you.

```rust
 use synfx_dsp_jit::build::*;
 use synfx_dsp_jit::instant_compile_ast;

 let (ctx, mut fun) = instant_compile_ast(
     op_add(literal(11.0), var("in1"))
 ).expect("No compile error");

 fun.init(44100.0, None); // Sample rate and optional previous DSPFunction

 let (sig1, sig2, res) = fun.exec_2in_2out(31.0, 10.0);

 // The result should be 11.0 + 31.0 == 42.0
 assert!((res - 42.0).abs() < 0.0001);

 // Yes, unfortunately you need to explicitly free this.
 // Because DSPFunction might be handed around to other threads.
 ctx.borrow_mut().free();
```

### DSP JIT API Example

Here is a more detailed example how the API can be used with state tracking.

```rust
use synfx_dsp_jit::*;
use synfx_dsp_jit::build::*;


// First we need to get a standard library with callable primitives/nodes:
let lib = get_standard_library();

// Then we create a DSPNodeContext to track newly created stateful nodes.
// You need to preserve this context across multiple calls to JIT::new() and JIT::compile().
let ctx = DSPNodeContext::new_ref();

// Create a new JIT compiler instance for compiling. Yes, you need to create a new one
// for each time you compile a DSPFunction.
let jit = JIT::new(lib.clone(), ctx.clone());

// This example shows how to use persistent variables (starting with '*')
// to build a simple phase increment oscillator
let ast = stmts(&[
    assign("*phase", op_add(var("*phase"), op_mul(literal(440.0), var("israte")))),
    _if(
        op_gt(var("*phase"), literal(1.0)),
        assign("*phase", op_sub(var("*phase"), literal(1.0))),
        None,
    ),
    var("*phase"),
]);

let mut dsp_fun = jit.compile(ASTFun::new(ast)).expect("No compile error");

// Initialize the function after compiling. For proper state tracking
// you will need to provide the previous DSPFunction as second argument to `init` here:
dsp_fun.init(44100.0, None);

// Create some audio samples:
let mut out = vec![];
for i in 0..200 {
    let (_, _, ret) = dsp_fun.exec_2in_2out(0.0, 0.0);
    if i % 49 == 0 {
        out.push(ret);
    }
}

// Just to show that this phase clock works:
assert!((out[0] - 0.0099).abs() < 0.0001);
assert!((out[1] - 0.4988).abs() < 0.0001);
assert!((out[2] - 0.9877).abs() < 0.0001);
assert!((out[3] - 0.4766).abs() < 0.0001);

ctx.borrow_mut().free();
```

### DSP Engine API

The [synfx_dsp_jit::engine::CodeEngine] API is a convenience API for dealing with
an audio/real time thread. When you want to compile the function on some non real time thread
like a GUI or worker thread, and use the resulting DSP function in an audio thread to produce
audio samples.

```rust
use synfx_dsp_jit::engine::CodeEngine;
use synfx_dsp_jit::build::*;

// Create an engine:
let mut engine = CodeEngine::new_stdlib();

// Retrieve the backend:
let mut backend = engine.get_backend();

// This should actually be in some audio thread:
std::thread::spawn(move || {
    backend.set_sample_rate(44100.0);

    loop {
        backend.process_updates(); // Receive updates from the frontend

        // Generate some audio samples here:
        for frame in 0..64 {
            let (s1, s2, ret) = backend.process(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
    }
});

// Upload a new piece of code whenever you see fit:
engine.upload(call("sin", 1, &[literal(1.0)])).unwrap();

let mut not_done = true;
while not_done {
    // Call this regularily!!!!
    engine.query_returns();

    // Just for ending this example:
    not_done = false;
}
```


License: GPL-3.0-or-later
