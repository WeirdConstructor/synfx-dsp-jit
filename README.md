# synfx-dsp-jit

synfx-dsp-jit is a specialized JIT compiler for digital (audio) signal processing for Rust.

This library allows you to compile an simplified abstract syntax tree (AST) down to machine code.
This crate uses the [Cranelift JIT compiler](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift)
for this task.

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

You can practically build a JIT compiled [Pure Data](https://puredata.info/)
or [SuperCollider](https://supercollider.github.io/) with this. Other notable
projects in this direction are:

- BitWig's "The Grid", which seems to use LLVM under the hood, either to AOT compiler the devices
or even JIT compiling the Grid itself (I'm not sure about that).
- [Gammou - polyphonic modular sound synthesizer](https://github.com/aliefhooghe/Gammou)



License: GPL-3.0-or-later
