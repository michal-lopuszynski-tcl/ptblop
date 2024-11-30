import copy

import pytest
import torch

import ptblop


def get_num_params(m: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


# Test unpruned model


def check_qwen_unpruned_forward(
    make_model_tokenizer_unprued_bp_config_fn,
    device: torch.device,
) -> None:
    model, tokenizer, bp_config0 = make_model_tokenizer_unprued_bp_config_fn()

    idx = tokenizer("How are you today?", return_tensors="pt")["input_ids"].to(device)
    model.to(device)

    with torch.no_grad():
        output1 = model(idx).logits

    ptblop.apply_bp_config_in_place(model, bp_config0)

    with torch.no_grad():
        output2 = model(idx).logits
        delta = torch.max(torch.abs(output1 - output2))

    assert delta.item() < 1.0e-5
    bp_config_test = ptblop.get_bp_config(model)
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


def check_qwen_disabled_attentnions(
    make_model_tokenizer_unpruned_bp_config_fn, device: torch.device
) -> None:
    model, tokenizer, bp_config0 = make_model_tokenizer_unpruned_bp_config_fn()

    idx = tokenizer("How are you today?", return_tensors="pt")["input_ids"].to(device)
    model.to(device)
    bp_config = make_bp_config_with_disabled_test_attentions(bp_config0)
    num_params0 = get_num_params(model)
    ptblop.apply_bp_config_in_place(model, bp_config)

    # Make sure forward works
    with torch.no_grad():
        _ = model(idx).logits

    bp_config_test = ptblop.get_bp_config(model)
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


def check_qwen_disabled_mlps(
    make_model_tokenizer_unpruned_bp_config_fn, device: torch.device
) -> None:
    model, tokenizer, bp_config0 = make_model_tokenizer_unpruned_bp_config_fn()

    idx = tokenizer("How are you today?", return_tensors="pt")["input_ids"].to(device)
    model.to(device)
    bp_config = make_bp_config_with_disabled_test_mlps(bp_config0)
    num_params0 = get_num_params(model)
    ptblop.apply_bp_config_in_place(model, bp_config)
    num_params1 = get_num_params(model)

    # Make sure forward works
    with torch.no_grad():
        _ = model(idx).logits

    bp_config_test = ptblop.get_bp_config(model)
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


def check_qwen_disabled_blocks(
    make_model_tokenizer_unpruned_bp_config_fn, device: torch.device
) -> None:
    model, tokenizer, bp_config0 = make_model_tokenizer_unpruned_bp_config_fn()

    idx = tokenizer("How are you today?", return_tensors="pt")["input_ids"].to(device)
    model.to(device)
    bp_config = make_bp_config_with_disabled_test_blocks(bp_config0)
    num_params0 = get_num_params(model)
    ptblop.apply_bp_config_in_place(model, bp_config)
    num_params1 = get_num_params(model)

    # Make sure forward works
    with torch.no_grad():
        _ = model(idx).logits

    bp_config_test = ptblop.get_bp_config(model)
    assert bp_config == bp_config_test

    assert num_params1 < num_params0
    assert ptblop.get_num_attention_blocks(model) == len(bp_config0) - 2
    assert ptblop.get_num_mlp_blocks(model) == len(bp_config0) - 2

    with pytest.raises(ValueError) as exc_info:
        ptblop.apply_bp_config_in_place(model, bp_config0)

    assert str(exc_info.value) == "Attention is used, but was set to None previously"
