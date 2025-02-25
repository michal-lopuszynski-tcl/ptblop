import copy

import ptblop
import pytest
import torch
import transformers

from ptblopgen.utils import get_num_active_params

try:
    import awq
except ModuleNotFoundError:
    awq = None


def make_model(model_name):

    # AWQ models work only with CUDA

    # model_name = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision = "main"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # Tokenizer and gen_data might be useful in the future

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, revision=model_revision, trust_remote_code=True
    )

    def __gen_data_qwen() -> torch.Tensor:
        return tokenizer("How are you today?", return_tensors="pt")["input_ids"]

    bp_config = ptblop.get_unpruned_bp_config(model)
    return model, __gen_data_qwen, bp_config


def test_get_num_active_params():
    model_bf16, _, bp_config0 = make_model("Qwen/Qwen2.5-0.5B-Instruct")
    p0_bf16 = ptblop.get_num_active_params(model_bf16)
    p0_bf6_ = get_num_active_params(model_bf16)
    assert p0_bf16 == p0_bf6_

    bpc = copy.deepcopy(bp_config0)
    bpc["model.layers.4"]["use_attention"] = False
    bpc["model.layers.4"]["use_mlp"] = False
    ptblop.apply_bp_config_in_place(model_bf16, bpc, set_unused_layers_to_none=False)

    p1_bf16 = ptblop.get_num_active_params(model_bf16)
    p1_bf6_ = get_num_active_params(model_bf16)
    assert p1_bf16 == p1_bf6_

    del model_bf16
    model_bf16, _, bp_config0 = make_model("Qwen/Qwen2.5-0.5B-Instruct")
    ptblop.apply_bp_config_in_place(model_bf16, bpc, set_unused_layers_to_none=True)
    p2_bf16 = ptblop.get_num_active_params(model_bf16)
    p2_bf16_ = get_num_active_params(model_bf16)

    assert p2_bf16_ == p2_bf16
    assert p2_bf16_ == p1_bf16


@pytest.mark.skipif(awq is None, reason="awq not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_get_num_active_params_awq():
    model_awq, _, bp_config0 = make_model("Qwen/Qwen2.5-0.5B-Instruct-AWQ")
    model_bf16, _, bp_config0 = make_model("Qwen/Qwen2.5-0.5B-Instruct")

    p0_bf16 = ptblop.get_num_active_params(model_bf16)
    p0_awq = get_num_active_params(model_bf16)
    assert p0_bf16 == p0_awq

    bpc = copy.deepcopy(bp_config0)
    bpc["model.layers.4"]["use_attention"] = False
    bpc["model.layers.4"]["use_mlp"] = False

    ptblop.apply_bp_config_in_place(model_bf16, bpc, set_unused_layers_to_none=False)
    ptblop.apply_bp_config_in_place(model_awq, bpc, set_unused_layers_to_none=False)

    p1_bf16 = ptblop.get_num_active_params(model_bf16)
    p1_awq = get_num_active_params(model_awq)
    assert p1_bf16 == p1_awq
