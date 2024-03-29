0.6.2 (2024-01-04)
==================

* Bump `synfx-dsp` dependency version to 0.5.6 (fixing a SIMD error).
* Change: Bumped cranelift version to 0.103 and updated the code accordingly.

0.6.1 (2022-11-06)
==================

* Bump `synfx-dsp` dependency version to 0.5.5

0.6.0 (2022-08-28)
==================

* Documentation: Added in [crate::stdlib] a rough documentation of the available variables and nodes.
* Change: Refactored the use of auxilary variables. Sample rate variables are not
prefixed with an `$` (`$srate` and `$israte`).
* Change: Persistent variables will now also be resetted to 0.0 after a reset or first initialization.
* Change: The `phase` node got a reset input now.
* Feature: Added reading/writing into a shared buffer of atomic floats.
See also `DSPNodeContext::atom` and `CodeEngine::atom`. Also added an example
that shows how to control a DSPFunction with atoms.
* Feature: Added constants for PI, E, TAU and many others.
* Feature: Added `$reset` which is true directly after an (explicit) reset and the first initialization.
* Feature: Added `s&h` and `s&h~` sample and hold nodes.
* Feature: Added buffers with AST operations for writing and reading the buffer contents.
Aswell as an AST operation for declaring the buffer size.
* Feature: Added fixed read only tables with f32 samples that can be swapped out from the
frontend. They are similar to the buffers, but they are not writeable. They are perfect for
shared pieces of audio samples.

0.5.3 (2022-08-05)
==================

* Bugfix: There was a bug in persistent variable handling, when there were
less persistent variables in the new DSPFunction.
* Bugfix: I've implemented now Sync too for the DSPFunction. Even though it's not
safe to do that. But it is required by some plugin APIs.
* Bugfix: Assigned variables would not be recognized and declared as local variables.
* Feature: Added documentation functions to the `DSPNodeType` trait. Also made it
mandatory for the macro helper to define a `DSPNodeType`.
* Feature: Added access to persistent variables via DSPFunction::access\_persistent\_var
and DSPNodeContext::get\_persistent\_variable\_index\_by\_name.
* Feature: Added stateless\_dsp\_node\_type.
* Feature: Added '/%' node to stdlib.
* Feature: Added 'phase' node to generate a phase sawtooth signal.
* Feature: Added CodeEngine API for compiling and uploading pieces of code to an audio thread.
* Feature: Added debug information collection to dump the Cranelift IR and AST tree.
* Documentation: Added example 'cpal\_jit\_dsp' to the repository.

0.5.2 (2022-07-31)
==================

* Feature: Made `walk_ast` public.
* Documentation: Minor tweaks.
* Feature: Multiple return values of DSP nodes can now be accessed via variables "%1" to "%5".
In the `DSPNodeType` trait these can be specified using the `DSPNodeSigBit::MultReturnPtr`.
Or via "M" in the `stateful_dsp_node_type` macro.

0.5.1 (2022-07-30)
==================

* Documentation: Fixes.


0.5.0 (2022-07-30)
==================

* Initial release.
