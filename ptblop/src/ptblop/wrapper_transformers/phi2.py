import logging
from typing import Any, Optional

import torch
import transformers  # type: ignore

from .. import prunable_block
from . import common

logger = logging.getLogger(__name__)


class PrunablePhi2BLock(torch.nn.Module, prunable_block.PrunableBlock):

    def get_unused_layer_names(self) -> set[str]:
        unused_layer_names = set()
        if not self.use_attention:
            unused_layer_names.add("self_attn")
        if not self.use_mlp:
            unused_layer_names.add("mlp")
        if not self.use_attention and not self.use_mlp:
            unused_layer_names.add("resid_dropout")
        return unused_layer_names

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
        self.self_attn = original_module.self_attn
        self.mlp = original_module.mlp
        self.input_layernorm = original_module.input_layernorm
        self.resid_dropout = original_module.resid_dropout
        # Called at the end sice it sets the unused layers to None automatically
        prunable_block.PrunableBlock.__init__(
            self,
            original_module=original_module,
            use_attention=use_attention,
            use_mlp=use_mlp,
            set_unused_layers_to_none=set_unused_layers_to_none,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, Any]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if self.use_attention:
            attn_outputs, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            attn_outputs = self.resid_dropout(attn_outputs)
        else:
            self_attn_weights = None

        if self.use_mlp:
            feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))

        if self.use_attention and self.use_mlp:
            result = attn_outputs + feed_forward_hidden_states + residual
        elif self.use_attention:
            result = attn_outputs + residual
        elif self.use_mlp:
            result = feed_forward_hidden_states + residual
        else:
            result = residual

        if output_attentions:
            return result, self_attn_weights
        else:
            return (result,)
