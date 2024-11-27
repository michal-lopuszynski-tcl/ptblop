import ptblop
import pytest
import torch
import transformers


def make_model_tokenizer_undecomposed_bp_config():
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


def test_unknown_modules():
    model, _, _ = make_model_tokenizer_undecomposed_bp_config()
    with pytest.raises(ValueError) as exc_info:
        ptblop.apply_bp_config_in_place(
            model,
            {
                "a": {"use_attention": True, "use_mlp": True},
                "b": {"use_attention": True, "use_mlp": True},
            },
        )
    assert str(exc_info.value) == "Unknown bp_config entries: a, b"
