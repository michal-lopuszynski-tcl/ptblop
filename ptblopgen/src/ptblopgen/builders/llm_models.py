import logging

import torch
import transformers

logger = logging.getLogger(__name__)


def _conv_str_to_dtype(s: str) -> torch.dtype:
    if s == "torch.float32":
        return torch.float32
    elif s == "torch.bfloat16":
        return torch.bfloat16
    elif s == "torch.float16":
        return torch.float16
    raise ValueError(f"Unknown dtype {s}")


def _add_pad_token(
    model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, model_name: str
) -> None:
    if (
        model_name
        in (
            "microsoft/phi-2",
            "upstage/SOLAR-10.7B-v1.0",
            "mistralai/Mistral-7B-Instruct-v0.2",
        )
        or model_name.startswith("meta-llama/Llama-2-")
        or model_name.startswith("meta-llama/Meta-Llama-3-")
        or model_name.startswith("meta-llama/Meta-Llama-3.1-")
        or model_name.startswith("Qwen/Qwen1.5-")
        or model_name.startswith("Qwen/Qwen2-")
        or model_name.startswith("Qwen/Qwen2.5-")
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # Phi-2 and LLama2 models don't have a pad token by default
        model.config.pad_token_id = tokenizer.pad_token_id  # llama, phi
        logger.info("Setting pad_token to eos_token")

    elif model_name == "Qwen/Qwen-1_8B":
        # See "https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Setting pad_token to <|endoftext|>")


def make_model(
    model_config,
    device: torch.device,
) -> tuple[torch.nn.Module, transformers.PreTrainedTokenizer]:
    model_name = model_config["model_name"]
    assert model_name.startswith("transformers/")
    model_name = model_name[13:]

    model_revision = model_config["model_revision"]
    model_dtype = _conv_str_to_dtype(model_config["model_dtype"])
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, revision=model_revision, trust_remote_code=True
    )
    _add_pad_token(model=model, model_name=model_name, tokenizer=tokenizer)
    model.to(model_dtype)
    model.to(device)
    return model, tokenizer
