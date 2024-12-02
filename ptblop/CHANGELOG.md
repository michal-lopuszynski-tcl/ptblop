# TODO
+ Add model.eval in tests
+ Add test if block with disabled both attention and mlps is equivallent to identity
+ Add logging in deleting modules
+ Add delete unused modules function
+ Review models and make sure all unused modules are nullified
+ Add `del original_module` in `apply_bp_config_in_place`
+ Rewrite everything in core.py using named_modules() (simplify)
+ Switch to ruff for linting and formatting
+ In `apply_bp_config_in_place`, how to call `fix_root_model` when multiple types of blocks are present? Should we care?

# DONE
+ Add test_cpu target to Makefile.inc
+ Add full block tests
+ Add vision transformer from timm
