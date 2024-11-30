import helpers
import pytest
import torch
import transformers  # type: ignore

import ptblop


def make_qwen_tokenizer_unpruned_bp_config() -> tuple[
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


def test_qwen_unpruned_forward_cpu() -> None:
    helpers.check_qwen_unpruned_forward(
        make_qwen_tokenizer_unpruned_bp_config, torch.device("cpu")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen_unpruned_forward_gpu() -> None:
    helpers.check_qwen_unpruned_forward(
        make_qwen_tokenizer_unpruned_bp_config, torch.device("cuda")
    )


def test_qwen_decomposed1_cpu() -> None:
    helpers.check_qwen_disabled_attentnions(
        make_qwen_tokenizer_unpruned_bp_config, torch.device("cpu")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen_decomposed1_gpu() -> None:
    helpers.check_qwen_disabled_attentnions(
        make_qwen_tokenizer_unpruned_bp_config, torch.device("cuda")
    )


def test_qwen_disabled_mlps_cpu() -> None:
    helpers.check_qwen_disabled_mlps(
        make_qwen_tokenizer_unpruned_bp_config, torch.device("cpu")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen_disabled_mlps_gpu() -> None:
    helpers.check_qwen_disabled_mlps(
        make_qwen_tokenizer_unpruned_bp_config, torch.device("cuda")
    )


def test_qwen_disabled_blocks_cpu() -> None:
    helpers.check_qwen_disabled_blocks(
        make_qwen_tokenizer_unpruned_bp_config, torch.device("cpu")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_qwen_disabled_blocks_gpu() -> None:
    helpers.check_qwen_disabled_blocks(
        make_qwen_tokenizer_unpruned_bp_config, torch.device("cuda")
    )
