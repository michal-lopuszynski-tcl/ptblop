import logging

from concurrent.futures import ThreadPoolExecutor
from typing import List

logger = logging.getLogger(__name__)


EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]


def extra_eos_for_direct_completion(dataset) -> List[str]:
    if dataset.lower() == "humaneval":
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    elif dataset.lower() == "mbpp":
        return ['\n"""', "\nassert"]
    raise ValueError(f"Unknown dataset: {dataset}")


# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def make_raw_chat_prompt(
    task_prompt: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer,
    enable_thinking=None,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template is None:
        return task_prompt

    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"

    task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
    if enable_thinking is None:
        apply_chat_template_kwargs = {"tokenize":False}
    else:
        apply_chat_template_kwargs = {
            "tokenize":False, "enable_thinking": enable_thinking
        }
    # logger.info(f"{enable_thinking=}")
    # logger.info(f"{apply_chat_template_kwargs=}")
    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response},
        ],
        **apply_chat_template_kwargs,
    ).split(_MAGIC_SPLITTER_)[0]
    return task_prompt

    # # For models with thinking enabled
    # task_prompt = tokenizer.apply_chat_template(
    #     [
    #         {"role": "user", "content": task_prompt},
    #         #{"role": "assistant", "content": response},
    #     ],
    #     **apply_chat_template_kwargs, add_generation_prompt=True,
    # )
    # logger.info(f"{task_prompt=}")
    # return task_prompt.split(_MAGIC_SPLITTER_)[0]


def concurrent_call(n, callback, /, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(callback, *args, **kwargs) for _ in range(n)]
        return [future.result() for future in futures]
