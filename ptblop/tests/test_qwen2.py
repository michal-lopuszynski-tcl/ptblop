import copy

import pytest
import torch
import transformers  # type: ignore

import ptblop


def get_num_params(m: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def make_model_tokenizer_unpruned_bp_config() -> tuple[
    transformers.PreTrainedModel,
    transformers.PreTrainedTokenizer,
    dict[str, dict[str, bool]],
]:
    model_name = "Qwen/Qwen2-0.5B"
    model_revision = "main"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, revision=model_revision, trust_remote_code=True
    )
    bp_config = ptblop.get_unpruned_bp_config(model)
    return model, tokenizer, bp_config


# Test unpruned forward


def check_qwen_unpruned_forward(device: torch.device) -> None:
    model, tokenizer, bp_config0 = make_model_tokenizer_unpruned_bp_config()

    idx = tokenizer("How are you today?", return_tensors="pt")["input_ids"].to(device)
    model.to(device)

    with torch.no_grad():
        output1 = model(idx).logits

    ptblop.apply_bp_config_in_place(model, bp_config0)

    with torch.no_grad():
        output2 = model(idx).logits
        delta = torch.max(torch.abs(output1 - output2))

    assert delta.item() < 1.0e-5


def test_qwen_unpruned_forward_cpu() -> None:
    check_qwen_unpruned_forward(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen_unpruned_forward_gpu() -> None:
    check_qwen_unpruned_forward(torch.device("cuda"))


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


def check_qwen_disabled_attentnions(device: torch.device) -> None:
    model, tokenizer, bp_config0 = make_model_tokenizer_unpruned_bp_config()

    idx = tokenizer("How are you today?", return_tensors="pt")["input_ids"].to(device)
    model.to(device)
    bp_config = make_bp_config_with_disabled_test_attentions(bp_config0)
    num_params0 = get_num_params(model)
    ptblop.apply_bp_config_in_place(model, bp_config)
    num_params1 = get_num_params(model)

    # Make sure forward works
    with torch.no_grad():
        _ = model(idx).logits

    assert num_params1 < num_params0
    assert ptblop.get_num_attention_blocks(model) == len(bp_config0) - 2
    assert ptblop.get_num_mlp_blocks(model) == len(bp_config0)

    with pytest.raises(ValueError) as exc_info:
        ptblop.apply_bp_config_in_place(model, bp_config0)

    assert str(exc_info.value) == "Attention is used, but was set to None previously"


def test_qwen_decomposed1_cpu() -> None:
    check_qwen_disabled_attentnions(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen_decomposed1_gpu() -> None:
    check_qwen_disabled_attentnions(torch.device("cuda"))


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


def check_qwen_disabled_mlps(device: torch.device) -> None:
    model, tokenizer, bp_config0 = make_model_tokenizer_unpruned_bp_config()

    idx = tokenizer("How are you today?", return_tensors="pt")["input_ids"].to(device)
    model.to(device)
    bp_config = make_bp_config_with_disabled_test_mlps(bp_config0)
    num_params0 = get_num_params(model)
    ptblop.apply_bp_config_in_place(model, bp_config)
    num_params1 = get_num_params(model)

    # Make sure forward works
    with torch.no_grad():
        _ = model(idx).logits

    assert num_params1 < num_params0
    assert ptblop.get_num_attention_blocks(model) == len(bp_config0)
    assert ptblop.get_num_mlp_blocks(model) == len(bp_config0) - 2

    with pytest.raises(ValueError) as exc_info:
        ptblop.apply_bp_config_in_place(model, bp_config0)

    assert str(exc_info.value) == "MLP is used, but was set to None previously"


def test_qwen_disabled_mlps_cpu() -> None:
    check_qwen_disabled_mlps(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen_disabled_mlps_gpu() -> None:
    check_qwen_disabled_mlps(torch.device("cuda"))
