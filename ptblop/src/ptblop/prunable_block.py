from typing import Optional

import abc

import torch


class PrunableBlock(abc.ABC):

    def __init__(self, use_attention: bool, use_mlp: bool):
        self.use_attention = use_attention
        self.use_mlp = use_mlp

    @abc.abstractmethod
    def set_unused_layers_to_none(self) -> None:
        pass

    @abc.abstractmethod
    def check_used_layers_not_none(self) -> None:
        pass

    @classmethod
    def fix_root_model(cls, root_model: torch.nn.Module) -> None:
        pass
