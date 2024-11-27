import torch

from .. import prunable_block


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
        if not isinstance(layer, prunable_block.PrunableBlock):
            counter += 1
            layer.self_attn.layer_idx = counter
        else:
            if layer.use_attention:
                layer.self_attn.layer_idx = counter
                counter += 1
