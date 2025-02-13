import abc
import logging

import torch

from . import utils

logger = logging.getLogger(__name__)


class PrunableBlock(abc.ABC):

    def __init__(
        self,
        original_module: torch.nn.Module,
        use_attention: bool,
        use_mlp: bool,
        set_unused_layers_to_none: bool,
    ):
        self.original_layer_names = [
            name for name, _ in original_module.named_children()
        ]
        self.use_attention = use_attention
        self.use_mlp = use_mlp
        if set_unused_layers_to_none:
            self.set_unused_layers_to_none()

    @abc.abstractmethod
    def get_unused_layer_names(self) -> set[str]:
        return set()

    def set_unused_layers_to_none(self) -> None:
        for layer_name in self.get_unused_layer_names():
            if hasattr(self, layer_name):
                setattr(self, layer_name, None)
                logger.info(f"Setting {layer_name} to None")
            else:
                msg = f"{utils.get_type_name(self)} has no atrr {layer_name}"
                raise ValueError(msg)

    def check_used_layers_not_none(self) -> None:
        unused_layer_names = self.get_unused_layer_names()

        for layer_name in self.original_layer_names:
            if layer_name not in unused_layer_names:
                if hasattr(self, layer_name):
                    layer = getattr(self, layer_name)
                    if layer is None:
                        full_layer_name = f"{utils.get_type_name(self)}.{layer_name}"
                        msg = f"{full_layer_name} is used, but was set to None before"
                        raise ValueError(msg)
                else:
                    msg = f"{utils.get_type_name(self)} has no atrr {layer_name}"
                    raise ValueError(msg)

    @classmethod
    def fix_root_model(cls, root_model: torch.nn.Module) -> None:
        pass
