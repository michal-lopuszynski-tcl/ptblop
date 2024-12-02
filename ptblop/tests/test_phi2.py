import helpers
import pytest
import torch
import transformers  # type: ignore

import ptblop


def make_phi2() -> helpers.MODEL_DATA_TYPE:
    model_name = "microsoft/phi-2"
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


def test_phi2_unpruned_forward_cpu() -> None:
    helpers.check_unpruned_forward(make_phi2, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_phi2_unpruned_forward_gpu() -> None:
    helpers.check_unpruned_forward(make_phi2, torch.device("cuda"))


def test_phi2_decomposed1_cpu() -> None:
    helpers.check_disabled_attentnions(make_phi2, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_phi2_decomposed1_gpu() -> None:
    helpers.check_disabled_attentnions(make_phi2, torch.device("cuda"))


def test_phi2_disabled_mlps_cpu() -> None:
    helpers.check_disabled_mlps(make_phi2, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_phi2_disabled_mlps_gpu() -> None:
    helpers.check_disabled_mlps(make_phi2, torch.device("cuda"))


def test_phi2_disabled_blocks_cpu() -> None:
    helpers.check_disabled_blocks(make_phi2, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_phi2_disabled_blocks_gpu() -> None:
    helpers.check_disabled_blocks(make_phi2, torch.device("cuda"))
