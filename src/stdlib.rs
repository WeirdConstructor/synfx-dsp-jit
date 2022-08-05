// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

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
    "A very simple oscillator that outputs a rising sawtooth wave with the \
    frequency 'freq' (range 0.0 to 22050.0)."
    inputs
    0 "freq"
    outputs
    0 "p"
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
    lib
}
