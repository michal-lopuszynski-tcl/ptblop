from collections.abc import Callable
from typing import Any

import torch

EVALUATOR_FN_TYPE = Callable[[torch.nn.Module, torch.device], dict[str, Any]]


def _make_model_and_evaluator_llm(
    model_config, evaluator_config, device
) -> tuple[torch.nn.Module, EVALUATOR_FN_TYPE]:
    from . import llm_evaluators, llm_models

    model, tokenizer = llm_models.make_model(model_config, device)
    evaluator_fn = llm_evaluators.make_evaluator(evaluator_config, tokenizer)
    return model, evaluator_fn


def _make_model_and_evaluator_vis(
    model_config, evaluator_config, device
) -> tuple[torch.nn.Module, EVALUATOR_FN_TYPE]:
    raise NotImplementedError()


def make_model_and_evaluator(
    model_config, evaluator_config, device
) -> tuple[torch.nn.Module, EVALUATOR_FN_TYPE]:
    model_name = model_config["model_name"]
    if model_name.startswith("transformers/"):
        return _make_model_and_evaluator_llm(model_config, evaluator_config, device)
    elif model_name.startswith("timm/"):
        return _make_model_and_evaluator_vis(model_config, evaluator_config, device)
    else:
        raise ValueError(f"Unsupported {model_name}")
