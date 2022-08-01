// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

use crate::{DSPNodeSigBit, DSPNodeType, DSPNodeTypeLibrary, DSPState};
use std::cell::RefCell;
use std::rc::Rc;

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

crate::stateful_dsp_node_type! {
    AccumNodeType, AccumNodeState => process_accum_nod "accum" "vSr"
    doc
    "This is a simple accumulator. It sums up it's input and returns it. \
     You can reset it's state if you pass a value >= 0.5 into 'reset'."
    inputs
    0 "input"
    1 "reset"
    outputs
    0 "sum"
}

#[derive(Default)]
struct SinNodeType;

extern "C" fn jit_sin(v: f64) -> f64 {
    v.sin()
}

impl DSPNodeType for SinNodeType {
    fn name(&self) -> &str {
        "sin"
    }

    fn function_ptr(&self) -> *const u8 {
        jit_sin as *const u8
    }

    fn signature(&self, i: usize) -> Option<DSPNodeSigBit> {
        match i {
            0 => Some(DSPNodeSigBit::Value),
            _ => None,
        }
    }

    fn has_return_value(&self) -> bool {
        true
    }
}

/// Creates a [crate::context::DSPNodeTypeLibrary] that contains a bunch of
/// standard components as seem fit by the synfx-dsp-jit crate developer.
///
/// Keep in mind, that you can always create your own custom library.
/// Or extend this one using the [DSPNodeType] trait.
pub fn get_standard_library() -> Rc<RefCell<DSPNodeTypeLibrary>> {
    let lib = Rc::new(RefCell::new(DSPNodeTypeLibrary::new()));
    lib.borrow_mut().add(Rc::new(SinNodeType::default()));
    lib.borrow_mut().add(AccumNodeType::new_ref());
    lib
}
