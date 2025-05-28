import timm
import torch


def make_model(model_config, device) -> torch.nn.Module:
    model_name = model_config["model_name"]
    assert model_name.startswith("timm/")
    model_name = model_name[5:]
    model = timm.create_model(model_name, pretrained=True)
    model.to(device)
    return model
