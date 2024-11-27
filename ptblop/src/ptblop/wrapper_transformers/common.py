import torch

from .. import prunable_block


def _set_self_attn_layer_idx(
    layer: torch.nn.Module | prunable_block.PrunableBlock, val: int
) -> None:
    self_attn = getattr(layer, "self_attn", None)
    if self_attn is not None:
        if hasattr(self_attn, "layer_idx"):
            self_attn.layer_idx = val
        else:
            msg = f"no `layer.self_attn.layer_idx` - unsuported layer {type(layer)}"
            raise ValueError(msg)
    else:
        raise ValueError(f"no `layer.self_attn` - unsuported layer {type(layer)}")


def fix_root_model_attention_indices(root_model: torch.nn.Module) -> None:
    # Oh, weird - but this is because of transormers model.model
    root_model_model = getattr(root_model, "model", None)
    if root_model_model is None:
        msg = "Unsupported model type -  `model.model` does not exist"
        raise ValueError(msg)

    root_layers = getattr(root_model_model, "layers", None)
    if root_layers is None:
        msg = "Unsupported model type -  `model.model.layers` does not exist"
        raise ValueError(msg)

    counter = 0
    for layer in root_layers:
        if isinstance(layer, prunable_block.PrunableBlock):
            if layer.use_attention:
                _set_self_attn_layer_idx(layer, counter)
                counter += 1
        else:
            _set_self_attn_layer_idx(layer, counter)
            counter += 1
