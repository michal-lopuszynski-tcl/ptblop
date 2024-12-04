import copy
from collections.abc import Callable

import pytest
import torch

import ptblop

MODEL_DATA_TYPE = tuple[
    torch.nn.Module, Callable[[], torch.Tensor], dict[str, dict[str, bool]]
]


def get_num_params(m: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def _forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    res = model(x)
    if hasattr(res, "logits"):
        return res.logits
    else:
        return res


# Test unpruned model


def check_unpruned_forward(
    make_model_fn: Callable[[], MODEL_DATA_TYPE],
    device: torch.device,
) -> None:
    model, gen_data, bp_config0 = make_model_fn()
    model.eval()
    assert len(bp_config0) > 0

    idx = gen_data().to(device)

    model.to(device)

    with torch.no_grad():
        output1 = _forward(model, idx)

    ptblop.apply_bp_config_in_place(model, bp_config0, set_unused_layers_to_none=True)

    with torch.no_grad():
        output2 = _forward(model, idx)
        delta = torch.max(torch.abs(output1 - output2))

    assert delta.item() < 1.0e-5

    bp_config_test = ptblop.get_bp_config(model)
    assert len(bp_config_test) > 0
    assert bp_config0 == bp_config_test


# Test pruned attentions


def make_bp_config_with_disabled_test_attentions(
    bp_config0: dict[str, dict[str, bool]]
) -> dict[str, dict[str, bool]]:
    bp_config = copy.deepcopy(bp_config0)
    ki = iter(bp_config.keys())
    _ = next(ki)
    k2 = next(ki)
    k3 = next(ki)
    bp_config[k2]["use_attention"] = False
    bp_config[k3]["use_attention"] = False
    return bp_config


def check_disabled_attentnions(
    make_model_fn: Callable[[], MODEL_DATA_TYPE], device: torch.device
) -> None:

    model, gen_data, bp_config0 = make_model_fn()
    model.eval()
    assert len(bp_config0) > 0

    idx = gen_data().to(device)
    model.to(device)
    bp_config = make_bp_config_with_disabled_test_attentions(bp_config0)
    num_params0 = get_num_params(model)
    ptblop.apply_bp_config_in_place(model, bp_config, set_unused_layers_to_none=True)

    # Make sure forward works
    with torch.no_grad():
        _ = _forward(model, idx)

    bp_config_test = ptblop.get_bp_config(model)
    assert len(bp_config_test) > 0
    assert bp_config == bp_config_test

    num_params1 = get_num_params(model)

    assert num_params1 < num_params0
    assert ptblop.get_num_attention_blocks(model) == len(bp_config0) - 2
    assert ptblop.get_num_mlp_blocks(model) == len(bp_config0)

    with pytest.raises(ValueError) as exc_info:
        ptblop.apply_bp_config_in_place(model, bp_config0)

    assert str(exc_info.value) == "Attention is used, but was set to None previously"


# Test pruned mlp


def make_bp_config_with_disabled_test_mlps(
    bp_config0: dict[str, dict[str, bool]]
) -> dict[str, dict[str, bool]]:
    bp_config = copy.deepcopy(bp_config0)
    ki = iter(bp_config.keys())
    _ = next(ki)
    k2 = next(ki)
    k3 = next(ki)
    bp_config[k2]["use_mlp"] = False
    bp_config[k3]["use_mlp"] = False
    return bp_config


def check_disabled_mlps(
    make_model_fn: Callable[[], MODEL_DATA_TYPE], device: torch.device
) -> None:
    model, gen_data, bp_config0 = make_model_fn()
    model.eval()
    assert len(bp_config0) > 0

    idx = gen_data().to(device)
    model.to(device)
    bp_config = make_bp_config_with_disabled_test_mlps(bp_config0)
    num_params0 = get_num_params(model)
    ptblop.apply_bp_config_in_place(model, bp_config, set_unused_layers_to_none=True)
    num_params1 = get_num_params(model)

    # Make sure forward works
    with torch.no_grad():
        _ = _forward(model, idx)

    bp_config_test = ptblop.get_bp_config(model)
    assert len(bp_config_test) > 0
    assert bp_config == bp_config_test

    assert num_params1 < num_params0
    assert ptblop.get_num_attention_blocks(model) == len(bp_config0)
    assert ptblop.get_num_mlp_blocks(model) == len(bp_config0) - 2

    with pytest.raises(ValueError) as exc_info:
        ptblop.apply_bp_config_in_place(model, bp_config0)

    assert str(exc_info.value) == "MLP is used, but was set to None previously"


# Test pruned blocks


def make_bp_config_with_disabled_test_blocks(
    bp_config0: dict[str, dict[str, bool]]
) -> dict[str, dict[str, bool]]:
    bp_config = copy.deepcopy(bp_config0)
    ki = iter(bp_config.keys())
    _ = next(ki)
    k2 = next(ki)
    k3 = next(ki)
    bp_config[k2]["use_attention"] = False
    bp_config[k3]["use_attention"] = False
    bp_config[k2]["use_mlp"] = False
    bp_config[k3]["use_mlp"] = False

    return bp_config


def check_disabled_blocks(
    make_model_fn: Callable[[], MODEL_DATA_TYPE], device: torch.device
) -> None:
    model, gen_data, bp_config0 = make_model_fn()
    model.eval()
    assert len(bp_config0) > 0

    idx = gen_data().to(device)
    model.to(device)
    bp_config = make_bp_config_with_disabled_test_blocks(bp_config0)
    num_params0 = get_num_params(model)
    ptblop.apply_bp_config_in_place(model, bp_config, set_unused_layers_to_none=True)
    num_params1 = get_num_params(model)

    # Make sure forward works
    with torch.no_grad():
        _ = _forward(model, idx)

    bp_config_test = ptblop.get_bp_config(model)
    assert len(bp_config_test) > 0
    assert bp_config == bp_config_test

    assert num_params1 < num_params0
    assert ptblop.get_num_attention_blocks(model) == len(bp_config0) - 2
    assert ptblop.get_num_mlp_blocks(model) == len(bp_config0) - 2

    with pytest.raises(ValueError) as exc_info:
        ptblop.apply_bp_config_in_place(
            model, bp_config0, set_unused_layers_to_none=True
        )

    assert str(exc_info.value) == "Attention is used, but was set to None previously"


def make_bp_config_pruning_enable_disable(
    bp_config0: dict[str, dict[str, bool]]
) -> dict[str, dict[str, bool]]:
    bp_config = copy.deepcopy(bp_config0)
    ki = iter(bp_config.keys())
    _ = next(ki)
    k2 = next(ki)
    k3 = next(ki)
    bp_config[k2]["use_attention"] = False
    bp_config[k3]["use_attention"] = False
    bp_config[k2]["use_mlp"] = False
    bp_config[k3]["use_mlp"] = False
    return bp_config


def check_enable_disable(
    make_model_fn: Callable[[], MODEL_DATA_TYPE],
    device: torch.device,
) -> None:
    model, gen_data, bp_config0 = make_model_fn()
    model.eval()
    assert len(bp_config0) > 0

    idx = gen_data().to(device)

    # 1. Run unmodified model
    model.to(device)
    with torch.no_grad():
        output1 = _forward(model, idx)

    # 2. Apply modified config - this should change the output

    bp_config = make_bp_config_pruning_enable_disable(bp_config0)
    ptblop.apply_bp_config_in_place(model, bp_config, set_unused_layers_to_none=False)

    bp_config_test = ptblop.get_bp_config(model)
    assert len(bp_config_test) > 0
    assert bp_config == bp_config_test

    with torch.no_grad():
        output2 = _forward(model, idx)
        delta2 = torch.max(torch.abs(output1 - output2))
    assert delta2.item() > 1.0e-5

    # 3. Apply original config - this should give the same output as original model

    ptblop.apply_bp_config_in_place(model, bp_config0, set_unused_layers_to_none=False)

    with torch.no_grad():
        output3 = _forward(model, idx)
        delta3 = torch.max(torch.abs(output1 - output3))
    assert delta3.item() < 1.0e-5


def make_bp_config_num_params(
    bp_config0: dict[str, dict[str, bool]]
) -> dict[str, dict[str, bool]]:
    bp_config = copy.deepcopy(bp_config0)
    ki = iter(bp_config.keys())
    _ = next(ki)
    k2 = next(ki)
    k3 = next(ki)
    k4 = next(ki)

    bp_config[k2]["use_attention"] = False
    bp_config[k2]["use_mlp"] = False

    bp_config[k3]["use_attention"] = False
    bp_config[k3]["use_mlp"] = True

    bp_config[k4]["use_attention"] = True
    bp_config[k4]["use_mlp"] = False

    return bp_config


def check_num_params(
    make_model_fn: Callable[[], MODEL_DATA_TYPE],
) -> None:
    model, _, bp_config0 = make_model_fn()
    model.eval()
    assert len(bp_config0) > 0

    params_orig = get_num_params(model)

    bp_config = make_bp_config_num_params(bp_config0)
    ptblop.apply_bp_config_in_place(model, bp_config, set_unused_layers_to_none=False)

    params_orig1 = get_num_params(model)
    params_prun1 = ptblop.get_num_params(model)

    ptblop.apply_bp_config_in_place(model, bp_config, set_unused_layers_to_none=True)
    params_prun2 = get_num_params(model)
    params_prun3 = ptblop.get_num_params(model)

    msg = f"{params_orig=}, {params_orig1=},"
    assert params_orig1 == params_orig
    msg = f"{params_prun1=}, {params_prun2=}, {params_prun3=}"
    assert params_prun1 == params_prun2 == params_prun3, msg
    msg = f"{params_prun1=}, {params_orig=}"
    assert params_prun1 < params_orig, msg