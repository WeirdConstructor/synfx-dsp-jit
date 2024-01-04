// Copyright (c) 2021-2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

use synfx_dsp_jit::*;

pub struct TSTState {
    pub l: f64,
    pub srate: f64,
}

impl TSTState {
    pub fn new() -> Self {
        Self { l: 0.0, srate: 0.0 }
    }

    pub fn reset(&mut self, dsp_state: &mut DSPState) {
        self.srate = dsp_state.srate;
    }
}

impl Default for TSTState {
    fn default() -> Self {
        Self::new()
    }
}

pub extern "C" fn test(x: f64, state: *mut DSPState, mystate: *mut u8, retvars: *mut [f64; 5]) -> f64 {
    unsafe {
        let p = mystate as *mut TSTState;
        (*state).x = x * 22.0;
        (*state).y = (*p).l;
        (*retvars)[0] = 90.4;
        (*retvars)[1] = 91.3;
        (*retvars)[2] = 92.2;
        (*retvars)[3] = 93.1;
        (*retvars)[4] = 94.0;
    };
    x * 10000.0 + 1.0
}

#[derive(Default)]
pub struct TestNodeType;

impl DSPNodeType for TestNodeType {
    fn name(&self) -> &str {
        "test"
    }
    fn function_ptr(&self) -> *const u8 {
        test as *const u8
    }

    fn signature(&self, i: usize) -> Option<DSPNodeSigBit> {
        match i {
            0 => Some(DSPNodeSigBit::Value),
            1 => Some(DSPNodeSigBit::DSPStatePtr),
            2 => Some(DSPNodeSigBit::NodeStatePtr),
            3 => Some(DSPNodeSigBit::MultReturnPtr),
            _ => None,
        }
    }

    fn has_return_value(&self) -> bool {
        true
    }

    fn reset_state(&self, dsp_state: *mut DSPState, state_ptr: *mut u8) {
        let ptr = state_ptr as *mut TSTState;
        unsafe {
            (*ptr).reset(&mut (*dsp_state));
        }
    }

    fn allocate_state(&self) -> Option<*mut u8> {
        Some(Box::into_raw(Box::new(TSTState::default())) as *mut u8)
    }

    fn deallocate_state(&self, ptr: *mut u8) {
        let _ = unsafe { Box::from_raw(ptr as *mut TSTState) };
    }
}
