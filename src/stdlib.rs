// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

/*! Definition of the Intermediate Representation for the JIT compiler.

This is a quick guide of the intermediate language you can pass into the JIT compiler
in form of an [crate::ASTNode] tree.

For a reference of the AST itself I recommend looking at [crate::ASTNode] and [crate::ASTBinOp].
They are pretty self explanatory.

Make sure to visit [crate::ast::build] too, it is an easy way to build an AST data structure.

## Global Variables

There are multiple different kinds of variables you can access:

- Predefined [crate::DSPFunction] Parameters (available in every function):
  - `in1` - First channel input
  - `in2` - Second channel input
  - `alpha` - Alpha parameter input
  - `beta` - Beta parameter input
  - `delta` - Delta parameter input
  - `gamma` - Gamma parameter input
  - `&sig1` - Writeable signal channel 1 output
  - `&sig2` - Writeable signal channel 2 output
- Global Constants
  - `PI` - 3.14159...
  - `TAU` - 2*PI
  - `E` - Eulers number
  - `1PI` - 1/PI
  - `2PI` - 2/PI
  - `PI2` - PI/2
  - `PI3` - PI/3
  - `PI4` - PI/4
  - `PI6` - PI/6
  - `PI8` - PI/8
  - `1SQRT2` - 1/sqrt(2)
  - `1SQRT_PI` - 1/sqrt(PI)
  - `LN2` - Ln(2)
  - `LN10` - Ln(10)
- Global Variables / Auxilary Variables:
  - `$srate` - The current sample rate in Hz
  - `$israte` - The current sample rate in 1.0 / Hz
- Persistent Variables (persistent across multiple calls, until a reset):
  - `*...` - Any variable that starts with a `*` is stored across multiple calls.
- Multiple Return Value Accessors
  - First return value does not exist as variable, it just is the result of
    the AST node itself, or the result of the node function.
  - `%1` - Second return value
  - `%2` - Third return value
  - `%3` - Fourth return value
  - `%4` - Fifth return value
  - `%5` - Sixth return value

## DSP Nodes

I heavily recommend checking out [HexoSynth](https://github.com/WeirdConstructor/HexoSynth)
if you plan to use `synfx-dsp-jit`, it offers a graphical environment for trying out
all the following nodes in real time in a visual programming language (called WBlockDSP):

| DSPNodeType name | Inputs | Outputs | Description |
|-|-|-|-|
| accum     | input, reset      | sum       | Accumulator, sums up the input |
| phase     | frequency, reset  | phase     | Phase oscillator |
| sin       | radians           | sine      | Sine function |
| /%        | a, b              | div, rem  | Computes the float division and remainder of a and b |
| atomr     | index             | value     | Reads an atomic float from a shared buffer |
| atomr~    | index             | value     | Reads a linear interpolated atomic float from a shared buffer |
| atomw     | index, value      | value     | Writes an atomic float into a shared buffer |

*/


use crate::stateful_dsp_node_type;
use crate::stateless_dsp_node_type;
use crate::{DSPNodeSigBit, DSPNodeType, DSPNodeTypeLibrary, DSPState};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

pub struct AccumNodeState {
    pub value: f64,
}

impl AccumNodeState {
    fn reset(&mut self, _state: &mut DSPState) {
        *self = Self::default();
    }
}

impl Default for AccumNodeState {
    fn default() -> Self {
        Self { value: 0.0 }
    }
}

extern "C" fn process_accum_nod(v: f64, r: f64, state: *mut AccumNodeState) -> f64 {
    let mut state = unsafe { &mut *state };
    if r > 0.5 {
        state.value = 0.0;
    } else {
        state.value += v;
    }
    state.value
}

stateful_dsp_node_type! {
    AccumNodeType, AccumNodeState => process_accum_nod "accum" "vvSr"
    doc
    "This is a simple accumulator. It sums up it's input and returns it. \
     You can reset it's state if you pass a value >= 0.5 into 'reset'."
    inputs
    0 "input"
    1 "reset"
    outputs
    0 "sum"
}

pub struct PhaseNodeState {
    pub israte: f64,
    pub value: f64,
}

impl PhaseNodeState {
    fn reset(&mut self, state: &mut DSPState) {
        *self = Self::default();
        self.israte = state.israte;
    }
}

impl Default for PhaseNodeState {
    fn default() -> Self {
        Self { israte: 1.0 / 44100.0, value: 0.0 }
    }
}

extern "C" fn process_phase(freq: f64, state: *mut PhaseNodeState) -> f64 {
    let mut state = unsafe { &mut *state };
    state.value += freq * state.israte;
    state.value = state.value.fract();
    state.value
}

stateful_dsp_node_type! {
    PhaseNodeType, PhaseNodeState => process_phase "phase" "vSr"
    doc
    "A very simple oscillator that outputs a rising sawtooth wave to 'phase' (range 0.0 to 1.0) with the \
    frequency 'freq' (range 0.0 to 22050.0)."
    inputs
    0 "freq"
    outputs
    0 "phase"
}

extern "C" fn process_sin(v: f64) -> f64 {
    v.sin()
}

stateless_dsp_node_type! {
    SinNodeType => process_sin "sin" "vr"
    doc
    "This is a sine function. Input is in radians."
    inputs
    0 ""
    outputs
    0 ""
}

extern "C" fn process_divrem(a: f64, b: f64, retvars: *mut [f64; 5]) -> f64 {
    unsafe {
        (*retvars)[0] = a % b;
    }
    a / b
}

stateless_dsp_node_type! {
    DivRemNodeType => process_divrem "/%" "vvMr"
    doc
    "Computes the float division and remainder of a / b"
    inputs
    0 "a"
    1 "b"
    outputs
    0 "div"
    1 "rem"
}

extern "C" fn process_atomr(idx: f64, dsp_state: *mut DSPState) -> f64 {
    let atoms = unsafe { &mut (*dsp_state).atoms };
    let i = idx.floor() as usize % atoms.len();
    atoms[i].get() as f64
}

stateless_dsp_node_type! {
    AtomRNodeType => process_atomr "atomr" "vDr"
    doc
    "This node reads from the specified 'index' in the 512 long array of \
    shared atomic floats. If index >= 512, it will wrap around."
    inputs
    0 "index"
    outputs
    0 "value"
}

extern "C" fn process_atomr_lin(idx: f64, dsp_state: *mut DSPState) -> f64 {
    let atoms = unsafe { &mut (*dsp_state).atoms };
    let i1 = idx.floor() as usize % atoms.len();
    let i2 = (i1 + 1) % atoms.len();
    let f = idx.fract();
    let a = atoms[i1].get() as f64;
    let b = atoms[i2].get() as f64;
    a * (1.0 - f) + f * b
}

stateless_dsp_node_type! {
    AtomRLinNodeType => process_atomr_lin "atomr~" "vDr"
    doc
    "This node reads linearily interpolated from the specified 'index' in the 512 long array of \
    shared atomic floats. If index >= 512, it will wrap around."
    inputs
    0 "index"
    outputs
    0 "value"
}

extern "C" fn process_atomw(idx: f64, v: f64, dsp_state: *mut DSPState) -> f64 {
    let atoms = unsafe { &mut (*dsp_state).atoms };
    let i = idx.floor() as usize % atoms.len();
    atoms[i].set(v as f32);
    v
}

stateless_dsp_node_type! {
    AtomWNodeType => process_atomw "atomw" "vvDr"
    doc
    "This node writes 'vlue' to the specified 'index' in the 512 long array of \
    shared atomic floats. If index >= 512, it will wrap around. It passes 'value' through, \
    so that you can easily plug this node in some signal path."
    inputs
    0 "index"
    1 "value"
    outputs
    0 "value"
}

/// Creates a [crate::context::DSPNodeTypeLibrary] that contains a bunch of
/// standard components as seem fit by the synfx-dsp-jit crate developer.
///
/// Keep in mind, that you can always create your own custom library.
/// Or extend this one using the [DSPNodeType] trait.
pub fn get_standard_library() -> Rc<RefCell<DSPNodeTypeLibrary>> {
    let lib = Rc::new(RefCell::new(DSPNodeTypeLibrary::new()));
    lib.borrow_mut().add(Arc::new(SinNodeType::default()));
    lib.borrow_mut().add(AccumNodeType::new_ref());
    lib.borrow_mut().add(DivRemNodeType::new_ref());
    lib.borrow_mut().add(PhaseNodeType::new_ref());
    lib.borrow_mut().add(AtomWNodeType::new_ref());
    lib.borrow_mut().add(AtomRNodeType::new_ref());
    lib.borrow_mut().add(AtomRLinNodeType::new_ref());
    lib
}
