from typing import Any

import torch


def get_type_name(o: Any) -> str:
    to = type(o)
    return to.__module__ + "." + to.__name__


def _split_module_parent_child_name(target: str) -> tuple[str, str]:
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def replace_submodule_in_place(
    root_module: torch.nn.Module, submodule_name: str, new_submodule: torch.nn.Module
) -> None:
    parent_name, child_name = _split_module_parent_child_name(submodule_name)
    parent_module = root_module.get_submodule(parent_name)
    setattr(parent_module, child_name, new_submodule)
