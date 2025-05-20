import json
import logging
import os
from typing import Any, Dict, List, Optional

from .provider import DecoderBase
from .sanitize import sanitize

logger = logging.getLogger(__name__)


def codegen(
    model: DecoderBase,
    dataset: Dict,
    greedy=False,
    n_samples=1,
    id_range=None,
    resume=True,
):
    logger.info(f"Processing dataset {len(dataset)=}")
    # task2nexist = {}
    # if resume and target_path.endswith(".jsonl") and os.path.isfile(target_path):
    #     with open(target_path, "r") as f:
    #         for line in f:
    #             if not line.strip():
    #                 continue
    #             task_id = json.loads(line)["task_id"]
    #             task2nexist[task_id] = task2nexist.get(task_id, 0) + 1

    # if target_path.endswith(".jsonl"):
    #     raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
    # else:
    #     raw_target_path = target_path + ".raw"
    #     os.makedirs(target_path, exist_ok=True)

    # logger.info(f"Sanitized code outputs will be saved to {target_path}")
    # logger.info(f"Raw outputs will be saved to {raw_target_path}")

    results = []
    n_dataset = len(dataset)
    time_gen_total = 0

    for q, (task_id, task) in enumerate(dataset.items(), start=1):
        # if id_range is not None:
        #     id_num = int(task_id.split("/")[1])
        #     low, high = id_range
        #     if id_num < low or id_num >= high:
        #         logger.info(f"Skipping {task_id} as it is not in {id_range}")
        #         continue

        # if not target_path.endswith(".jsonl"):
        #     p_name = task_id.replace("/", "_")
        #     os.makedirs(os.path.join(target_path, p_name), exist_ok=True)
        #     task2nexist[task_id] = len(
        #         [
        #             f
        #             for f in os.listdir(os.path.join(target_path, p_name))
        #             if f.endswith(".py")
        #         ]
        #     )

        n_more_samples = n_samples

        # if resume and task2nexist.get(task_id, 0) > 0:
        #     log += f" (resuming from {task2nexist[task_id]})"
        #     n_more_samples -= task2nexist[task_id]

        log = f"Codegen for {task_id}, {q} of {n_dataset}"
        logger.info(log)

        sidx = n_samples - n_more_samples
        while sidx < n_samples:
            prompt = task["prompt"].strip() + "\n"
            # RUNNING MODEL IS HERE
            g = model.codegen(
                prompt=prompt,
                do_sample=not greedy,
                num_samples=n_samples - sidx,
            )
            time_gen_total += g["time_gen"]
            outputs = g["outputs"]
            assert outputs, "No outputs from model!"
            for impl in outputs:
                solution = prompt + impl if model.is_direct_completion() else impl
                sanitized_solution = sanitize(solution, entrypoint=task["entry_point"])

                r = {
                    "task_id": task_id,
                    "solution": sanitized_solution,
                    "prompt_raw": g["prompt_raw"],
                    "outputs_raw": g["outputs_raw"],
                    "time_gen": g["time_gen"],
                }

                results.append(r)
                # if target_path.endswith(".jsonl"):
                #     # Writing the sanitized version
                #     with open(target_path, "a") as f:
                #         f.write(
                #             json.dumps(
                #                 {"task_id": task_id, "solution": sanitized_solution}
                #             )
                #             + "\n"
                #         )

                #     # Writing the raw version
                #     with open(raw_target_path, "a") as f:
                #         f.write(
                #             json.dumps({"task_id": task_id, "solution": solution})
                #             + "\n"
                #         )

                # else:
                #     # Writing the sanitized version
                #     with open(
                #         os.path.join(target_path, p_name, f"{sidx}.py"),
                #         "w",
                #         encoding="utf-8",
                #     ) as f:
                #         f.write(sanitized_solution)

                #     # Writing the raw version
                #     with open(
                #         os.path.join(raw_target_path, p_name, f"{sidx}.py"),
                #         "w",
                #         encoding="utf-8",
                #     ) as f:
                #         f.write(solution)
                sidx += 1
    logger.info(f"{time_gen_total=:.2f} seconds")
    return results


def get_type_name(o: Any) -> str:
    return type(o).__module__ + "." + type(o).__name__


def make_model_runner(
    *,
    model_name,
    model,
    tokenizer,
    dataset,
    temperature,
    force_base_prompt,
    instruction_prefix,
    response_prefix,
    enable_thinking,
    max_new_tokens,
):
    model_type_name = get_type_name(model)
    if model_type_name.startswith("transformers."):
        logger.info("Detected HF model")
        from .provider import hf

        hf_decoder_kwargs = {
            "name": model_name,
            "model": model,
            "tokenizer": tokenizer,
            "dataset": dataset,
            "tempearture": temperature,
            "force_base_prompt": force_base_prompt,
            "instruction_prefix": instruction_prefix,
            "response_prefix": response_prefix,
            "enable_thinking": enable_thinking,
        }
        if max_new_tokens is not None:
            hf_decoder_kwargs["max_new_tokens"] = max_new_tokens
        return hf.HuggingFaceDecoder(**hf_decoder_kwargs)
    else:
        raise ValueError(f"Unsupported model type {model_type_name}")


def run_codegen(
    # model_name: str,
    model,
    tokenizer,
    dataset: str,
    dataset_dict: Dict,
    root: str = "tmp/results",
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    resume: bool = True,
    greedy: bool = False,
    id_range: List = None,
    backend: str = "vllm",
    force_base_prompt: bool = False,
    evalperf_type: str = None,  # For EvalPerf
    jsonl_fmt: bool = True,
    enable_thinking=None,
    max_new_tokens=None,
):
    assert dataset in ["humaneval", "mbpp", "evalperf"], f"Invalid dataset {dataset}"
    assert evalperf_type is None or evalperf_type in [
        "instruct",
        "perf-instruct",
        "perf-CoT",
    ]
    model_name = "model"
    # Make dir for codes generated by each model
    identifier = (
        model_name.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}"
    )
    if evalperf_type:
        identifier += f"-{evalperf_type}"

    target_path = os.path.join(root, dataset, identifier)
    if jsonl_fmt:
        target_path += ".jsonl"
    else:
        os.makedirs(target_path, exist_ok=True)

    all_tasks_complete = False
    if jsonl_fmt and os.path.isfile(target_path):
        task_counts = {}
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_id = data["task_id"]
                task_counts[task_id] = task_counts.get(task_id, 0) + 1

            all_tasks_complete = all(
                task_counts.get(task_id, 0) >= n_samples
                for task_id in dataset_dict.keys()
            )

    if all_tasks_complete:
        logger.info("All samples are already cached. Skipping codegen.")
        return target_path

    if greedy and (temperature != 0 or bs != 1 or n_samples != 1):
        temperature = 0.0
        bs = 1
        n_samples = 1
        logger.info(
            "Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0"
        )

    if id_range is not None:
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    if bs is None:
        bs = min(n_samples, 32)
        logger.info(f"Setting batch size to {bs=}")

    # # Make project dir
    # os.makedirs(root, exist_ok=True)
    # # Make dataset dir
    # os.makedirs(os.path.join(root, dataset), exist_ok=True)

    # Model instructions
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

    if evalperf_type == "perf-instruct":
        instruction_prefix = "Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type == "perf-CoT":
        instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type is not None and evalperf_type != "instruct":
        raise ValueError(f"Invalid evalperf_type: {evalperf_type}")

    # Model creation
    model_runner = make_model_runner(
        model_name=model_name,
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        temperature=temperature,
        force_base_prompt=force_base_prompt,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
    )

    results = codegen(
        dataset=dataset_dict,
        greedy=greedy,
        model=model_runner,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
    )

    for i, sample in enumerate(results):
        assert (
            "completion" in sample or "solution" in sample
        ), "No completion or solution found in sample!"
        assert "solution" not in sample or isinstance(
            sample["solution"], str
        ), "Solution must be a string! If you have multiple solutions, please repeat the task_id."
        assert "completion" not in sample or isinstance(
            sample["completion"], str
        ), "Completion must be a string! If you have multiple solutions, please repeat the task_id."

        sample["_identifier"] = sample["task_id"] + f" (line {i+1} in memory)"
    results_dict = {sample["task_id"]: sample for sample in results}
    return results_dict
