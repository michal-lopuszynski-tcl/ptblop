from ._version import __version__, __version_info__  # noqa: F401
from .core import (  # noqa: F401
    apply_bp_config_in_place,
    get_bp_config,
    get_num_active_params,
    get_num_attention_blocks,
    get_num_mlp_blocks,
    get_num_prunable_blocks,
    get_unpruned_bp_config,
    has_prunable_blocks,
)
from .prunable_block import PrunableBlock  # noqa: F401
