import logging
from typing import Optional, cast

import torch

from . import prunable_block, utils

logger = logging.getLogger(__name__)

_WRAPPER_DICT_TYPE = dict[type[torch.nn.Module], type[prunable_block.PrunableBlock]]

try:
    import transformers  # type: ignore
    from transformers.models.llama.modeling_llama import (  # type: ignore
        LlamaDecoderLayer,
    )
    from transformers.models.phi.modeling_phi import PhiDecoderLayer  # type: ignore
    from transformers.models.qwen2.modeling_qwen2 import (  # type: ignore
        Qwen2DecoderLayer,
    )

    def is_old_tranformers() -> bool:
        version = transformers.__version__
        major, minor = map(int, version.split(".")[:2])
        return (major < 4) or (major == 4 and minor < 48)

    if is_old_tranformers():
        from .wrapper_transformers_pre448 import (
            PrunableLlamaBlock,
            PrunablePhi2BLock,
            PrunableQwen2Block,
        )
    else:
        from .wrapper_transformers import (  # type: ignore[assignment]
            PrunableLlamaBlock,
            PrunablePhi2BLock,
            PrunableQwen2Block,
        )

    _BLOCK_TYPE_TO_WRAPPER_TYPE_TRANSFORMERS: _WRAPPER_DICT_TYPE = {
        Qwen2DecoderLayer: PrunableQwen2Block,
        PhiDecoderLayer: PrunablePhi2BLock,
        LlamaDecoderLayer: PrunableLlamaBlock,
    }
except ImportError:
    _BLOCK_TYPE_TO_WRAPPER_TYPE_TRANSFORMERS = {}

try:
    import timm  # type: ignore

    from .wrapper_timm import PrunableVisionTransformerBlock

    _BLOCK_TYPE_TO_WRAPPER_TYPE_TIMM: _WRAPPER_DICT_TYPE = {
        timm.models.vision_transformer.Block: PrunableVisionTransformerBlock,
    }
except ImportError:
    _BLOCK_TYPE_TO_WRAPPER_TYPE_TIMM = {}

_BLOCK_TYPE_TO_WRAPPER_TYPE: _WRAPPER_DICT_TYPE = (
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


def get_unused_parameters(
    prunable_block: prunable_block.PrunableBlock,
) -> list[torch.nn.Parameter]:
    res = []
    for layer_name in prunable_block.get_unused_layer_names():
        msg = f"{layer_name=} not in {prunable_block}"
        assert hasattr(prunable_block, layer_name), msg
        layer = getattr(prunable_block, layer_name)
        if layer is not None:
            res.extend(list(layer.parameters()))
    return res


def get_num_active_params(module: torch.nn.Module, only_trainable: bool = False) -> int:
    unused_parameters = []
    for _, submodule in module.named_modules():
        if isinstance(submodule, prunable_block.PrunableBlock):
            unused_parameters.extend(get_unused_parameters(submodule))

    unused_data_ptrs = {p.data_ptr() for p in unused_parameters}

    parameters = list(module.parameters())

    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]

    unique = {
        p.data_ptr(): p for p in parameters if p.data_ptr() not in unused_data_ptrs
    }.values()

    return sum(p.numel() for p in unique)


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


def _is_model_on_multiple_gpus(model: torch.nn.Module) -> bool:
    devices = set()

    for param in model.parameters():
        if param.is_cuda:
            devices.add(param.get_device())

            if len(devices) > 1:
                return True

    return len(devices) > 1


def _redispatch(module: torch.nn.Module) -> None:
    try:
        import accelerate  # type: ignore

        if hasattr(module, "hf_device_map"):
            logger.info("Redispatching model using accelerate library")
            accelerate.dispatch_model(module, module.hf_device_map)
    except ImportError:
        pass


def apply_bp_config_in_place(
    module: torch.nn.Module,
    bp_config: dict[str, dict[str, bool]],
    *,
    set_unused_layers_to_none: bool = True,
    suppress_multigpu_redispatch: bool = False,
) -> None:
    config_entries = set(bp_config.keys())
    module_entries = {n for n, _ in module.named_modules()}
    unknown_entries = config_entries - module_entries

    if len(unknown_entries) > 0:
        unknown_entries_str = ", ".join(sorted(unknown_entries))
        raise ValueError(f"Unknown bp_config entries: {unknown_entries_str}")

    last_prunable_submodule: Optional[prunable_block.PrunableBlock] = None

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
                msg = f"Wraping {submodule_name} of {utils.get_type_name(submodule)}"
                logger.info(msg)
                wrapper_type = _BLOCK_TYPE_TO_WRAPPER_TYPE[submodule_type]
                if module_config is None:
                    module_config = {"use_attention": True, "use_mlp": True}
                new_submodule = wrapper_type(
                    original_module=submodule,
                    use_attention=module_config["use_attention"],
                    use_mlp=module_config["use_mlp"],
                    set_unused_layers_to_none=set_unused_layers_to_none,
                )
                utils.replace_submodule_in_place(
                    module, submodule_name, cast(torch.nn.Module, new_submodule)
                )
                last_prunable_submodule = new_submodule

    # Assumption - there should be only one type of modules
    # TODO: Perhaps add collecting all prunable types and call fix_root_model for each?
    if last_prunable_submodule is not None:
        last_prunable_submodule.fix_root_model(module)

    # Fix multigpu model after applying wrapping
    if (
        not suppress_multigpu_redispatch
        and torch.cuda.device_count() > 1
        and _is_model_on_multiple_gpus(module)
    ):
        _redispatch(module)
