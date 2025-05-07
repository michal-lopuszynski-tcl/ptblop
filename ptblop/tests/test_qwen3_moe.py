import os

import helpers
import pytest
import torch
import transformers  # type: ignore

import ptblop


def is_large_disabled() -> bool:
    return os.environ.get("TEST_PTBLOP_LARGE") is None


def is_qwen3_moe_available() -> bool:
    return hasattr(transformers.models, "qwen3_moe")


def make_qwen3_moe() -> helpers.MODEL_DATA_TYPE:
    model_name = "Qwen/Qwen3-30B-A3B"
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

    def __gen_data_qwen() -> torch.Tensor:
        return tokenizer("How are you today?", return_tensors="pt")["input_ids"]

    bp_config = ptblop.get_unpruned_bp_config(model)
    return model, __gen_data_qwen, bp_config


# # CPU tests take forever, enable this if smaller moe is released
# @pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
# @pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
# def test_qwen3_moe_unpruned_forward_cpu() -> None:
#     helpers.check_unpruned_forward(make_qwen3_moe, torch.device("cpu"))
#


@pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
@pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen3_moe_unpruned_forward_gpu() -> None:
    helpers.check_unpruned_forward(make_qwen3_moe, torch.device("cuda"))


# # CPU tests take forever, enable this if smaller moe is released
# @pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
# @pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
# def test_qwen3_moe_decomposed1_cpu() -> None:
#     helpers.check_disabled_attentnions(make_qwen3_moe, torch.device("cpu"))


@pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
@pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen3_moe_decomposed1_gpu() -> None:
    helpers.check_disabled_attentnions(make_qwen3_moe, torch.device("cuda"))


# # CPU tests take forever, enable this if smaller moe is released
# @pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
# @pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
# def test_qwen3_moe_disabled_mlps_cpu() -> None:
#     helpers.check_disabled_mlps(make_qwen3_moe, torch.device("cpu"))


@pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
@pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen3_moe_disabled_mlps_gpu() -> None:
    helpers.check_disabled_mlps(make_qwen3_moe, torch.device("cuda"))


# # CPU tests take forever, enable this if smaller moe is released
# @pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
# @pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
# def test_qwen3_moe_disabled_blocks_cpu() -> None:
#     helpers.check_disabled_blocks(make_qwen3_moe, torch.device("cpu"))
#


@pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
@pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen3_moe_disabled_blocks_gpu() -> None:
    helpers.check_disabled_blocks(make_qwen3_moe, torch.device("cuda"))


# # CPU tests take forever, enable this if smaller moe is released
# @pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
# @pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
# def test_qwen3_moe_enable_disable_cpu() -> None:
#     helpers.check_enable_disable(make_qwen3_moe, torch.device("cpu"))


@pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
@pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen3_moe_enable_disable_gpu() -> None:
    helpers.check_enable_disable(make_qwen3_moe, torch.device("cuda"))


# # CPU tests take forever, enable this if smaller moe is released
# @pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
# @pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
# def test_qwen3_moe_num_params() -> None:
#     helpers.check_num_params(make_qwen3_moe)

# # CPU tests take forever, enable this if smaller moe is released
# @pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
# @pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
# def test_qwen3_moe_disabled_block0_is_identity_cpu() -> None:
#    helpers.check_disabled_block_is_identity(
#        make_qwen3_moe, torch.device("cpu"), "model.layers.0", 0
#    )


@pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
@pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen3_moe_disabled_block0_is_identity_gpu() -> None:
    helpers.check_disabled_block_is_identity(
        make_qwen3_moe, torch.device("cuda"), "model.layers.0", 0
    )


# # CPU tests take forever, enable this if smaller moe is released
# @pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
# @pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
# def test_qwen3_moe_disabled_block5_is_identity_cpu() -> None:
#     helpers.check_disabled_block_is_identity(
#         make_qwen3_moe, torch.device("cpu"), "model.layers.5", 5
#     )


@pytest.mark.skipif(not is_qwen3_moe_available(), reason="qwen3 not available")
@pytest.mark.skipif(is_large_disabled(), reason="large tests not enabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen3_moe_disabled_block5_is_identity_gpu() -> None:
    helpers.check_disabled_block_is_identity(
        make_qwen3_moe, torch.device("cuda"), "model.layers.5", 5
    )
