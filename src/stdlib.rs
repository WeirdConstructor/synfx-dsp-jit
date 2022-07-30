// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

use crate::{DSPNodeTypeLibrary, DSPNodeType, DSPNodeSigBit};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct SinNodeType;

pub extern "C" fn jit_sin(v: f64) -> f64 {
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

/// Creates a [crate::jit::DSPNodeTypeLibrary] that contains a bunch of
/// standard components as seem fit by the synfx-dsp-jit crate developer.
///
/// Keep in mind, that you can always create your own custom library.
/// Or extend this one using the [DSPNodeType] trait.
pub fn get_standard_library() -> Rc<RefCell<DSPNodeTypeLibrary>> {
    let lib = Rc::new(RefCell::new(DSPNodeTypeLibrary::new()));
    lib.borrow_mut().add(Rc::new(SinNodeType::default()));
    lib
}
