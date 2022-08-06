0.6.0 (unreleased)
==================

* Documentation: Added in [crate::stdlib] a rough documentation of the available variables and nodes.
* Change: Refactored the use of auxilary variables. Sample rate variables are not
prefixed with an `$` (`$srate` and `$israte`).
* Feature: Added reading/writing into a shared buffer of atomic floats.
See also `DSPNodeContext::atom` and `CodeEngine::atom`. Also added an example
that shows how to control a DSPFunction with atoms.
* Feature: Added constants for PI, E, TAU and many others.

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
