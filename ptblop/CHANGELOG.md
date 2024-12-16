# TODO
+ **!!.** Test against qwen1.5-4B?
+ **!..** Add `set_unused_layers_to_none` function to `ptblop`
+ **!..** Define type alias for `bp_config`
+ **...** Add raising value error if the `bp_config` on `apply_bp_config_in_place` is malformed
+ **...** Add logging in deleting modules
+ **...** Rewrite everything in core.py using named_modules() (simplify)
+ **...** **[checks]** Switch to ruff for linting and formatting
+ **...** In `apply_bp_config_in_place`, how to call `fix_root_model` when multiple types of blocks are present? Should we care?
+ **...** Add some visualization of models with pruned blocks?

# DONE
+ **[tests]** Add test if block with disabled both attention and mlps is equivallent to identity
+ Review models and make sure all unused modules are nullified
+ Simplify implementation so the get unused module names is used more
+ Add parameter counting functionality
+ **[tests]** Add model.eval in tests
+ **[tests]** Add test_cpu target to Makefile.inc
+ **[tests]** Add full block tests
+ Add vision transformer from timm
