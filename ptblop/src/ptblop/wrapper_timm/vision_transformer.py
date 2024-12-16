from typing import TYPE_CHECKING

import timm  # type: ignore
import torch

from .. import prunable_block


class PrunableVisionTransformerBlock(torch.nn.Module, prunable_block.PrunableBlock):
    def get_unused_layer_names(self) -> set[str]:
        unused_layer_names = set()
        if not self.use_attention:
            unused_layer_names.add("attn")
            unused_layer_names.add("ls1")
            unused_layer_names.add("norm1")
            unused_layer_names.add("drop_path1")
        if not self.use_mlp:
            unused_layer_names.add("mlp")
            unused_layer_names.add("ls2")
            unused_layer_names.add("norm2")
            unused_layer_names.add("drop_path2")
        return unused_layer_names

    # def set_unused_layers_to_none(self) -> None:
    #     if not self.use_attention:
    #         self.attn = None
    #         self.ls1 = None
    #         self.norm1 = None
    #         self.drop_path1 = None

    #     if not self.use_mlp:
    #         self.mlp = None
    #         self.ls2 = None
    #         self.norm2 = None
    #         self.drop_path2 = None

    # def check_used_layers_not_none(self) -> None:
    #     if self.use_attention:
    #         if (
    #             self.attn is None
    #             or self.ls1 is None
    #             or self.norm1 is None
    #             or self.drop_path1 is None
    #         ):
    #             raise ValueError("Attention is used, but was set to None previously")
    #     if self.use_mlp:
    #         if (
    #             self.mlp is None
    #             or self.ls2 is None
    #             or self.norm2 is None
    #             or self.drop_path2 is None
    #         ):
    #             raise ValueError("MLP is used, but was set to None previously")

    def __init__(
        self,
        original_module: timm.models.vision_transformer.Block,
        use_attention: bool = True,
        use_mlp: bool = True,
        set_unused_layers_to_none: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        prunable_block.PrunableBlock.__init__(
            self,
            original_module=original_module,
            use_attention=use_attention,
            use_mlp=use_mlp,
        )

        self.norm1 = original_module.norm1
        self.attn = original_module.attn
        self.ls1 = original_module.ls1
        self.drop_path1 = original_module.drop_path1

        self.norm2 = original_module.norm2
        self.mlp = original_module.mlp
        self.ls2 = original_module.ls2
        self.drop_path2 = original_module.drop_path2

        if set_unused_layers_to_none:
            self.set_unused_layers_to_none()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            if TYPE_CHECKING:
                assert (
                    self.drop_path1 is not None
                    and self.ls1 is not None
                    and self.attn is not None
                    and self.norm1 is not None
                )
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        if self.use_mlp:
            if TYPE_CHECKING:
                assert (
                    self.drop_path2 is not None
                    and self.ls2 is not None
                    and self.mlp is not None
                    and self.norm2 is not None
                )
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
