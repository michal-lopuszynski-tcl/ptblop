from typing import TYPE_CHECKING, Any, Optional

import torch
import transformers  # type: ignore

from .. import prunable_block
from . import common

_FORWARD_OUTPUT_TYPE = (
    tuple[torch.Tensor]
    | tuple[torch.Tensor, Optional[torch.Tensor]]
    | tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
)


class PrunableLlamaBlock(torch.nn.Module, prunable_block.PrunableBlock):

    def get_unused_layer_names(self) -> list[str]:
        unused_layer_names = []
        if not self.use_attention:
            unused_layer_names.append("self_attn")
        if not self.use_mlp:
            unused_layer_names.append("mlp")
        return unused_layer_names

    def set_unused_layers_to_none(self) -> None:
        if not self.use_attention:
            self.self_attn = None
        if not self.use_mlp:
            self.mlp = None

    def check_used_layers_not_none(self) -> None:
        if self.use_attention and self.self_attn is None:
            raise ValueError("Attention is used, but was set to None previously")
        if self.use_mlp and self.mlp is None:
            raise ValueError("MLP is used, but was set to None previously")

    @classmethod
    def fix_root_model(cls, root_model: torch.nn.Module) -> None:
        common.fix_root_model_attention_indices(root_model)

    def __init__(
        self,
        original_module: transformers.models.llama.modeling_llama.LlamaDecoderLayer,
        use_attention: bool = True,
        use_mlp: bool = True,
        set_unused_layers_to_none: bool = True,
    ):
        torch.nn.Module.__init__(self)
        prunable_block.PrunableBlock.__init__(
            self, use_attention=use_attention, use_mlp=use_mlp
        )
        self.hidden_size = original_module.hidden_size
        self.self_attn = original_module.self_attn
        self.mlp = original_module.mlp
        self.input_layernorm = original_module.input_layernorm
        self.post_attention_layernorm = original_module.post_attention_layernorm
        if set_unused_layers_to_none:
            if not self.use_attention:
                self.self_attn = None
            if not self.use_mlp:
                self.mlp = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[transformers.Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> _FORWARD_OUTPUT_TYPE:

        if self.use_attention:
            if TYPE_CHECKING:
                assert self.self_attn is not None
            out = self.input_layernorm(hidden_states)
            out, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=out,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = hidden_states + out
        else:
            self_attn_weights = None
            present_key_value = past_key_value

        if self.use_mlp:
            if TYPE_CHECKING:
                assert self.mlp is not None
            out = self.post_attention_layernorm(hidden_states)
            out = self.mlp(out)
            hidden_states = hidden_states + out

        outputs: _FORWARD_OUTPUT_TYPE = (hidden_states,)

        if output_attentions:
            outputs = (*outputs, self_attn_weights)

        if use_cache:
            outputs = (*outputs, present_key_value)

        return outputs
