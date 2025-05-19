import logging
import time

from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import DecoderBase
from .utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)


logger = logging.getLogger(__name__)


class HuggingFaceDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        model,
        tokenizer,
        force_base_prompt: bool = False,
        tempearture: float = 0.0,
        enable_thinking = None,
        **kwargs,
    ):
        super().__init__(name=name, temperature=tempearture, **kwargs)
        self.device = model.device
        self.skip_special_tokens = True
        self.force_base_prompt = force_base_prompt
        self.tokenizer = tokenizer
        if self.is_direct_completion():  # no chat template
            self.eos += extra_eos_for_direct_completion(dataset)
        else:  # with chat template
            self.eos += ["\n```\n"]

        self.model = model
        self.model = self.model.to(self.device)
        self.enable_thinking = enable_thinking
        # # For models with thinking enabled
        # self.max_new_tokens = 4096
        logger.info(f"HF Wrapper: max_new_tokens={self.max_new_tokens}")
        logger.info(f"HF Wrapper: batch_size={self.batch_size}")
        logger.info(f"HF Wrapper: temperature={self.temperature}")
        logger.info(f"HF Wrapper: enable_thinking={self.enable_thinking}")
        logger.info(f"HF Wrapper: force_base_prompt={self.force_base_prompt}")
        logger.info(f"HF Wrapper: is_direct_completion={self.is_direct_completion()}")


    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.tokenizer, self.enable_thinking
            )
        )
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature
        t_start = time.perf_counter()
        with torch.inference_mode():
            self.model.eval()
            outputs = self.model.generate(
                input_tokens,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                num_return_sequences=min(self.batch_size, num_samples),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                stop_strings=self.eos,
                tokenizer=self.tokenizer,
                **kwargs,
            )
        time_gen = time.perf_counter() - t_start

        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )
        outputs_final = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs_final.append(output[:min_index].replace("\t", "    "))

        outputs_raw = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        r = {"outputs": outputs_final, "prompt_raw": prompt, "outputs_raw": outputs_raw, "time_gen": time_gen}
        return r
