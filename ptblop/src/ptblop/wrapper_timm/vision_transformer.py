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

    def __init__(
        self,
        original_module: timm.models.vision_transformer.Block,
        use_attention: bool = True,
        use_mlp: bool = True,
        set_unused_layers_to_none: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)

        self.norm1 = original_module.norm1
        self.attn = original_module.attn
        self.ls1 = original_module.ls1
        self.drop_path1 = original_module.drop_path1

        self.norm2 = original_module.norm2
        self.mlp = original_module.mlp
        self.ls2 = original_module.ls2
        self.drop_path2 = original_module.drop_path2

        # Called at the end sice it sets the unused layers to None automatically
        prunable_block.PrunableBlock.__init__(
            self,
            original_module=original_module,
            use_attention=use_attention,
            use_mlp=use_mlp,
            set_unused_layers_to_none=set_unused_layers_to_none,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        if self.use_mlp:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
