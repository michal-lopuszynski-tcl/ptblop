# TODO
+ Switch to ruff for linting and formatting
+ Handle orphaned layer-norm in case of disabled both attention and mlp
+ In `apply_bp_config_in_place`, how to call `fix_root_model` when multiple types of blocks are present? Should we care?
