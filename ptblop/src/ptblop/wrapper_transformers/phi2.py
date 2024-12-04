import logging
from typing import TYPE_CHECKING, Any, Optional

import torch
import transformers  # type: ignore

from .. import prunable_block
from . import common

logger = logging.getLogger(__name__)

_FORWARD_OUTPUT_TYPE = (
    tuple[torch.Tensor]
    | tuple[torch.Tensor, Optional[torch.Tensor]]
    | tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
)


class PrunablePhi2BLock(torch.nn.Module, prunable_block.PrunableBlock):

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
        original_module: transformers.models.phi.modeling_phi.PhiDecoderLayer,
        use_attention: bool = True,
        use_mlp: bool = True,
        set_unused_layers_to_none: bool = False,
    ):
        torch.nn.Module.__init__(self)
        prunable_block.PrunableBlock.__init__(
            self, use_attention=use_attention, use_mlp=use_mlp
        )
        self.self_attn = original_module.self_attn
        self.mlp = original_module.mlp
        self.input_layernorm = original_module.input_layernorm
        self.resid_dropout = original_module.resid_dropout
        if set_unused_layers_to_none:
            self.set_unused_layers_to_none()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> _FORWARD_OUTPUT_TYPE:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if self.use_attention:
            if TYPE_CHECKING:
                assert self.self_attn is not None
            attn_outputs, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            attn_outputs = self.resid_dropout(attn_outputs)
        else:
            self_attn_weights = None
            present_key_value = past_key_value

        if self.use_mlp:
            if TYPE_CHECKING:
                assert self.mlp is not None
            feed_forward_hidden_states = self.mlp(hidden_states)
            feed_forward_hidden_states = self.resid_dropout(feed_forward_hidden_states)

        if self.use_attention and self.use_mlp:
            result = attn_outputs + feed_forward_hidden_states + residual
        elif self.use_attention:
            result = attn_outputs + residual
        elif self.use_mlp:
            result = feed_forward_hidden_states + residual
        else:
            result = residual

        outputs: _FORWARD_OUTPUT_TYPE = (result,)

        if output_attentions:
            outputs = (*outputs, self_attn_weights)

        if use_cache:
            outputs = (*outputs, present_key_value)

        return outputs
