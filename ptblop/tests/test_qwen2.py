import helpers
import pytest
import torch
import transformers  # type: ignore

import ptblop


def make_qwen2() -> helpers.MODEL_DATA_TYPE:
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

    def __gen_data_qwen() -> torch.Tensor:
        return tokenizer("How are you today?", return_tensors="pt")["input_ids"]

    bp_config = ptblop.get_unpruned_bp_config(model)
    return model, __gen_data_qwen, bp_config


def test_qwen2_unpruned_forward_cpu() -> None:
    helpers.check_unpruned_forward(make_qwen2, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen2_unpruned_forward_gpu() -> None:
    helpers.check_unpruned_forward(make_qwen2, torch.device("cuda"))


def test_qwen2_decomposed1_cpu() -> None:
    helpers.check_disabled_attentnions(make_qwen2, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen2_decomposed1_gpu() -> None:
    helpers.check_disabled_attentnions(make_qwen2, torch.device("cuda"))


def test_qwen2_disabled_mlps_cpu() -> None:
    helpers.check_disabled_mlps(make_qwen2, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen2_disabled_mlps_gpu() -> None:
    helpers.check_disabled_mlps(make_qwen2, torch.device("cuda"))


def test_qwen2_disabled_blocks_cpu() -> None:
    helpers.check_disabled_blocks(make_qwen2, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen2_disabled_blocks_gpu() -> None:
    helpers.check_disabled_blocks(make_qwen2, torch.device("cuda"))


def test_qwen2_enable_disable_cpu() -> None:
    helpers.check_enable_disable(make_qwen2, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen2_enable_disable_gpu() -> None:
    helpers.check_enable_disable(make_qwen2, torch.device("cuda"))


def test_qwen2_num_params() -> None:
    helpers.check_num_params(make_qwen2)


def test_qwen2_disabled_block0_is_identity_cpu() -> None:
    helpers.check_disabled_block_is_identity(
        make_qwen2, torch.device("cpu"), "model.layers.0", 0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen2_disabled_block0_is_identity_gpu() -> None:
    helpers.check_disabled_block_is_identity(
        make_qwen2, torch.device("cuda"), "model.layers.0", 0
    )


def test_qwen2_disabled_block5_is_identity_cpu() -> None:
    helpers.check_disabled_block_is_identity(
        make_qwen2, torch.device("cpu"), "model.layers.5", 5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen2_disabled_block5_is_identity_gpu() -> None:
    helpers.check_disabled_block_is_identity(
        make_qwen2, torch.device("cuda"), "model.layers.5", 5
    )
