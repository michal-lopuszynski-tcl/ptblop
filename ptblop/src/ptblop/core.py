import logging

import torch

from . import prunable_block

logger = logging.getLogger(__name__)


try:
    from transformers.models.qwen2.modeling_qwen2 import (  # type: ignore
        Qwen2DecoderLayer,
    )

    from .wrapper_transformers import PrunableQwen2Block

    _BLOCK_TYPE_TO_WRAPPER_TYPE_TRANSFORMERS = {Qwen2DecoderLayer: PrunableQwen2Block}
except ImportError:
    _BLOCK_TYPE_TO_WRAPPER_TYPE_TRANSFORMERS = {}

try:
    import timm

    from .wrapper_timm import PrunableVisionTransformerBlock

    _BLOCK_TYPE_TO_WRAPPER_TYPE_TIMM = {
        timm.models.vision_transformer.Block: PrunableVisionTransformerBlock,
    }
except ImportError:
    _BLOCK_TYPE_TO_WRAPPER_TYPE_TIMM = {}

_BLOCK_TYPE_TO_WRAPPER_TYPE = (
    _BLOCK_TYPE_TO_WRAPPER_TYPE_TRANSFORMERS | _BLOCK_TYPE_TO_WRAPPER_TYPE_TIMM
)


def has_prunable_blocks(module: torch.nn.Module) -> int:
    for _, child_module in module.named_children():
        if isinstance(child_module, prunable_block.PrunableBlock):
            return True
        else:
            if has_prunable_blocks(child_module):
                return True
    return False


def get_num_prunable_blocks(module: torch.nn.Module) -> int:
    num_prunable_blocks = 0
    for _, child_module in module.named_children():
        if isinstance(child_module, prunable_block.PrunableBlock):
            num_prunable_blocks += 1
        else:
            num_prunable_blocks += get_num_prunable_blocks(child_module)
    return num_prunable_blocks


def get_num_attention_blocks(module: torch.nn.Module) -> int:
    num_attention = 0
    for _, child_module in module.named_children():
        if isinstance(child_module, prunable_block.PrunableBlock):
            num_attention += int(child_module.use_attention)
        else:
            num_attention += get_num_attention_blocks(child_module)
    return num_attention


def get_num_mlp_blocks(module: torch.nn.Module) -> int:
    num_attention = 0
    for _, child_module in module.named_children():
        if isinstance(child_module, prunable_block.PrunableBlock):
            num_attention += int(child_module.use_mlp)
        else:
            num_attention += get_num_mlp_blocks(child_module)
    return num_attention


def _get_prunable_block_bp_config(m: prunable_block.PrunableBlock) -> dict[str, bool]:
    return {
        "use_attention": m.use_attention,
        "use_mlp": m.use_mlp,
    }


def _get_bp_config(
    module: torch.nn.Module, module_path: str
) -> dict[str, dict[str, bool]]:
    res = {}
    for child_name, child_module in module.named_children():
        if module_path:
            prefix = module_path + "."
        else:
            prefix = module_path

        if isinstance(child_module, prunable_block.PrunableBlock):
            res[prefix + child_name] = _get_prunable_block_bp_config(child_module)
        else:
            child_config = _get_bp_config(child_module, prefix + child_name)
            res |= child_config
    return res


def get_bp_config(module: torch.nn.Module) -> dict[str, dict[str, bool]]:
    return _get_bp_config(module, "")


def get_unpruned_bp_config(module: torch.nn.Module) -> dict[str, dict[str, bool]]:
    res: dict[str, dict[str, bool]] = {}
    for submodule_name, submodule in module.named_modules():
        submodule_type = type(submodule)
        if submodule_type in _BLOCK_TYPE_TO_WRAPPER_TYPE or isinstance(
            submodule, prunable_block.PrunableBlock
        ):
            res[submodule_name] = {"use_attention": True, "use_mlp": True}
    return res


def _split_module_parent_child_name(target: str) -> tuple[str, str]:
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def _replace_submodule_in_place(
    root_module: torch.nn.Module, submodule_name: str, new_submodule: torch.nn.Module
) -> None:
    parent_name, child_name = _split_module_parent_child_name(submodule_name)
    parent_module = root_module.get_submodule(parent_name)
    setattr(parent_module, child_name, new_submodule)


def apply_bp_config_in_place(
    module: torch.nn.Module,
    bp_config: dict[str, dict[str, bool]],
    set_unused_layers_to_none: bool = True,
) -> None:
    config_entries = set(bp_config.keys())
    module_entries = {n for n, _ in module.named_modules()}
    unknown_entries = config_entries - module_entries

    if len(unknown_entries) > 0:
        unknown_entries_str = ", ".join(sorted(unknown_entries))
        raise ValueError(f"Unknown bp_config entries: {unknown_entries_str}")

    last_prunable_submodule = None

    for submodule_name, submodule in module.named_modules():
        module_config = bp_config.get(submodule_name)

        if isinstance(submodule, prunable_block.PrunableBlock):
            if module_config is not None:
                submodule.use_attention = module_config["use_attention"]
                submodule.use_mlp = module_config["use_mlp"]
                if set_unused_layers_to_none:
                    submodule.set_unused_layers_to_none()
                # Check if we did not enable layers previously set to None
                submodule.check_used_layers_not_none()
                last_prunable_submodule = submodule
        else:
            submodule_type = type(submodule)
            if submodule_type in _BLOCK_TYPE_TO_WRAPPER_TYPE:
                logger.info(f"Wraping {submodule_name} of {submodule_type}")
                wrapper_type = _BLOCK_TYPE_TO_WRAPPER_TYPE[submodule_type]
                if module_config is None:
                    module_config = {"use_attention": True, "use_mlp": True}
                new_submodule = wrapper_type(
                    original_module=submodule,
                    use_attention=module_config["use_attention"],
                    use_mlp=module_config["use_mlp"],
                    set_unused_layers_to_none=set_unused_layers_to_none,
                )
                _replace_submodule_in_place(module, submodule_name, new_submodule)
                last_prunable_submodule = new_submodule

    # Assumption - there should be only one type of modules
    # TODO: Perhaps add collecting all prunable types and call fix_root_model for each?
    if last_prunable_submodule is not None:
        last_prunable_submodule.fix_root_model(module)
