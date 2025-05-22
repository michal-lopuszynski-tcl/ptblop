import copy
import gzip
import hashlib
import importlib
import json
import logging
import multiprocessing
import os
import pickle
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import config
from .codegen import run_codegen
from .data.mbpp import mbpp_deserialize_inputs, mbpp_serialize_inputs
from .eval import (
    PASS,
    estimate_pass_at_k,
    untrusted_check,
)
from .eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[bool]]


logger = logging.getLogger(__name__)


def trusted_exec(code, inputs, entry_point, record_time=False, output_not_none=False):
    """Execute trusted code in place."""
    exec_globals = {}
    exec(code, exec_globals)
    fn = exec_globals[entry_point]

    rtime = []
    ret = []
    for inp in inputs:
        inp = copy.deepcopy(inp)
        if record_time:
            start = time.time()
            ret.append(fn(*inp))
            rtime.append(time.time() - start)
        else:
            ret.append(fn(*inp))

    if output_not_none:
        ret = [i is not None for i in ret]

    if record_time:
        return ret, rtime
    else:
        return ret


def get_groundtruth(*, dataset_name, dataset, dataset_hash, cache_dir_prefix):
    if dataset_name == "humaneval":
        tasks_only_output_not_none = []
    elif dataset_name == "mbpp":
        tasks_only_output_not_none = MBPP_OUTPUT_NOT_NONE_TASKS
    else:
        raise ValueError(f"Unknown {dataset_name=}")
    cache_dir = cache_dir_prefix + str(os.getpid())
    cache_file = os.path.join(cache_dir, f"{dataset_hash}.pkl")
    if os.path.exists(cache_file):
        logger.info(f"Load from ground-truth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Computing expected output...")
    tbegin = time.time()
    expected_output = {}
    for task_id, problem in dataset.items():
        oracle = {}
        oracle["base"], oracle["base_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )

        oracle["plus"], oracle["plus_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        expected_output[task_id] = oracle
    logger.info(f"Expected outputs computed in {time.time() - tbegin:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected_output, f)

    return expected_output


def check_correctness(
    dataset: str,
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    expected_output: Dict[str, List],
    base_only=False,
    fast_check=False,
    identifier=None,
    min_time_limit: float = config.DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = config.DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> Dict[str, Result]:  # {...}, "base" | "plus" -> (status, details)
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    ret["base"] = untrusted_check(
        dataset,
        solution,
        problem["base_input"],
        problem["entry_point"],
        expected=expected_output["base"],
        atol=problem["atol"],
        ref_time=expected_output["base_time"],
        fast_check=fast_check,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )

    if not base_only:
        ret["plus"] = untrusted_check(
            dataset,
            solution,
            problem["plus_input"],
            problem["entry_point"],
            expected=expected_output["plus"],
            atol=problem["atol"],
            ref_time=expected_output["plus_time"],
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )

    return ret


def load_jsonl_gz(f):
    res = {}
    for line in f:
        d = json.loads(line)
        res[d["task_id"]] = d
    return res


def get_dataset_dict(dataset_name, limit):
    if dataset_name not in ["mbpp", "humaneval"]:
        raise ValueError(f"Unknown {dataset_name=}")

    if dataset_name == "mbpp":
        fname = "MbppPlus-v0.2.0.jsonl.gz"
        pkg_ref = importlib.resources.files(__package__)
        data_file = pkg_ref / "resources" / fname
        logger.info(f"Loading dataset from {fname}")

        with data_file.open("rb") as f:
            with gzip.open(f, "rt", encoding="utf-8") as f_decompressed:
                dataset = load_jsonl_gz(f_decompressed)

        for task_id, task in dataset.items():
            task["base_input"] = mbpp_deserialize_inputs(task_id, task["base_input"])
            task["plus_input"] = mbpp_deserialize_inputs(task_id, task["plus_input"])
    elif dataset_name == "humaneval":
        fname = "HumanEvalPlus-v0.1.10.jsonl.gz"
        logger.info(f"Loading dataset from {fname}")
        pkg_ref = importlib.resources.files(__package__)
        data_file = pkg_ref / "resources" / fname
        with data_file.open("rb") as f:
            with gzip.open(f, "rt", encoding="utf-8") as f_decompressed:
                dataset = load_jsonl_gz(f_decompressed)
    else:
        raise ValueError(f"Unknown {dataset_name=}, but should never be raised")

    if limit is None:
        return dataset
    else:
        n1 = len(dataset)
        n2 = int(limit * n1)
        dataset = {k: v for i, (k, v) in enumerate(dataset.items()) if i < n2}
        n3 = len(dataset)
        logger.info(f"Truncating dataset from {fname} with {limit=}: {n1} -> {n3}")
        return dataset


def get_hash(problems):
    return hashlib.md5(repr(problems).encode("utf-8")).hexdigest()


def run_solutions_tests(
    *,
    dataset,
    dataset_solutions,
    expected_solutions,
    dataset_problems,
    parallel,
    base_only,
    test_details,
    min_time_limit,
    gt_time_limit_factor,
):
    n_workers = parallel or max(1, multiprocessing.cpu_count() // 2)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        eval_results = defaultdict(list)
        remainings = set()

        for sample in dataset_solutions.values():
            task_id = sample["task_id"]
            if task_id not in dataset_problems:
                logger.warning(
                    f"{task_id=} is found in the samples but not found in the dataset"
                )
                continue
            solution = (
                sample["solution"]
                if "solution" in sample
                else dataset_problems[task_id]["prompt"] + sample["completion"]
            )
            remainings.add(sample["_identifier"])
            args = (
                dataset,
                completion_id[task_id],
                dataset_problems[task_id],
                solution,
                expected_solutions[task_id],
                base_only,
                not test_details,  # fast_check
                sample["_identifier"],
                min_time_limit,
                gt_time_limit_factor,
            )
            futures.append(executor.submit(check_correctness, *args))
            completion_id[task_id] += 1
            n_samples += 1

        assert n_samples == len(remainings), "Missing problems in unfinished"
        assert len(completion_id) == len(
            dataset_problems
        ), "Missing problems in samples"

        def stucking_checker():
            while remainings:
                last_size = len(remainings)
                time.sleep(20)
                if last_size != len(remainings) or len(remainings) == 0:
                    continue
                # Potential stucking
                logger.warning("No samples had finished testing in the last 20s")
                logger.warning(f"{len(remainings)} samples to be tested: {remainings}")

        threading.Thread(target=stucking_checker).start()

        for future in as_completed(futures):
            result = future.result()
            remainings.remove(result["_identifier"])
            eval_results[result["task_id"]].append(result)
    return eval_results


def summarize_solutions(
    *, dataset, dataset_problems, eval_results, base_only, test_details
):
    eval_summary = {}

    # sort the results for each problem by completion_id
    for task_id, task_results in eval_results.items():
        task_results.sort(key=lambda x: x["completion_id"])
        eval_summary[task_id] = []
        for res in task_results:

            def get_failed_tests(stat, details, inputs) -> List[Any]:
                if stat == PASS or not details:
                    return []

                if test_details:
                    return [inputs[i] for i in range(len(details)) if not details[i]]

                # else => simply return the only and the last fail test
                return [inputs[len(details) - 1]]

            base_stat, base_details = res["base"]
            base_fail_tests = get_failed_tests(
                base_stat, base_details, dataset_problems[task_id]["base_input"]
            )

            # initialize plus tests
            plus_stat = None
            plus_fail_tests = []

            # with plus tests
            if not base_only:
                plus_stat, plus_details = res["plus"]
                plus_fail_tests = get_failed_tests(
                    plus_stat, plus_details, dataset_problems[task_id]["plus_input"]
                )

            if dataset == "mbpp":
                base_fail_tests = mbpp_serialize_inputs(task_id, base_fail_tests)
                plus_fail_tests = mbpp_serialize_inputs(task_id, plus_fail_tests)

            eval_summary[task_id].append(
                {
                    "task_id": task_id,
                    "solution": res["solution"],
                    "base_status": base_stat,
                    "plus_status": plus_stat,
                    "base_fail_tests": base_fail_tests,
                    "plus_fail_tests": plus_fail_tests,
                }
            )

    # Calculate pass@k.
    total = np.array([len(r) for r in eval_summary.values()])
    base_correct = []
    new_correct = []

    for res in eval_summary.values():
        bc = sum([r["base_status"] == PASS for r in res])
        base_correct.append(bc)
        if not base_only:
            new_correct.append(
                sum(
                    [
                        res[i]["base_status"] == res[i]["plus_status"] == PASS
                        for i in range(len(res))
                    ]
                )
            )
    base_correct = np.array(base_correct)

    base_pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean().item()
        for k in [1, 10, 100]
        if total.min() >= k
    }

    if new_correct:
        new_correct = np.array(new_correct)
        plus_pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, new_correct, k).mean().item()
            for k in [1, 10, 100]
            if (total >= k).all()
        }
    else:
        plus_pass_at_k = None

    return eval_summary, base_pass_at_k, plus_pass_at_k


def prepare_evaluate_results(
    *,
    dataset,
    dataset_hash,
    dataset_problems,
    dataset_solutions,
    eval_results,
    base_only,
    test_details,
):
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hash": dataset_hash,
        "eval": None,  # <- placeholder to keep the `eval` entry at the top
    }

    eval_summary, base_pass_at_k, plus_pass_at_k = summarize_solutions(
        dataset=dataset,
        dataset_problems=dataset_problems,
        eval_results=eval_results,
        base_only=base_only,
        test_details=test_details,
    )

    results["pass_at_k"] = {"base": base_pass_at_k}

    if not base_only:
        results["pass_at_k"]["plus"] = plus_pass_at_k

    for k, vs in eval_summary.items():
        for v in vs:
            v["prompt_raw"] = dataset_solutions[k]["prompt_raw"]
            v["outputs_raw"] = dataset_solutions[k]["outputs_raw"]
            v["time_gen"] = dataset_solutions[k]["time_gen"]

    results["eval"] = eval_summary
    return results


def split_problems(dataset_problems, n):
    dataset_ids = list(dataset_problems.keys())
    dataset_selected_ids = set(random.sample(dataset_ids, k=n))

    dataset_problems1 = {
        k: v for k, v in dataset_problems.items() if k in dataset_selected_ids
    }
    dataset_problems2 = {
        k: v for k, v in dataset_problems.items() if k not in dataset_selected_ids
    }
    assert len(dataset_problems1) == n
    assert len(dataset_problems1) + len(dataset_problems2) == len(dataset_problems)
    return dataset_problems1, dataset_problems2


def is_eligible_for_early_stopping(
    dataset_problems_early, dataset_solutions_early, eval_results_early
):
    t1 = time.perf_counter()
    assert set(dataset_problems_early.keys()) == set(dataset_solutions_early.keys())
    assert set(dataset_problems_early.keys()) == set(eval_results_early.keys())

    # The task is eligible for early stopping if in initial sample
    # 1. All test failed
    # 2. None of the solutions contains `entry_point`

    for task_id, task_data_list in eval_results_early.items():
        # eval_results_early["HumanEval/2"][0]["plus"]
        for j, task_data in enumerate(task_data_list, start=1):
            result = task_data["base"][0]

            assert result in {"pass", "fail"}, f"{result=} not pass/fail"
            if result == "pass":
                duration = time.perf_counter() - t1
                logger.info(
                    f"Disabling early stopping, {task_id}.{j} passed, "
                    f"check duration {duration:.2f} s.",
                )
                return False
            if "plus" in task_data:
                result = task_data["plus"][0]
                assert result in {"pass", "fail"}, f"{result=} not pass/fail"
                if result == "pass":
                    duration = time.perf_counter() - t1
                    logger.info(
                        f"Disabling early stopping, {task_id}.{j} passed, "
                        f"check duration {duration:.2f} s."
                    )
                    return False

    for task_id, solution in dataset_solutions_early.items():
        entry_point = dataset_problems_early[task_id]["entry_point"]

        n_prompt = len(solution["prompt_raw"])
        for j, output in enumerate(solution["outputs_raw"], start=1):
            output_generated = output[n_prompt:]
            logger.info("\n" + output_generated + "\n")
            if entry_point in output_generated:
                duration = time.perf_counter() - t1
                logger.info(
                    f"Disabling early stopping, {task_id}.{j} contains "
                    f"entry_point, check duration {duration:.2f} s."
                )
                return False
    duration = time.perf_counter() - t1
    logger.info("Eanbling early stopping, check duration {duration:.2f} s.")
    return True


def evaluate(
    *,
    model,
    tokenizer,
    dataset: str,
    cache_dir_prefix: str,
    base_only: bool = False,
    parallel: Optional[int] = None,
    test_details: bool = False,
    min_time_limit: float = config.DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = config.DEFAULT_GT_TIME_LIMIT_FACTOR,
    greedy: bool = True,
    enable_thinking: Optional[bool] = None,
    limit: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    n_early_stopping: Optional[int] = None,
    # **model_kwargs,
):
    t_start = time.perf_counter()

    dataset_problems = get_dataset_dict(dataset, limit)
    dataset_hash = get_hash(dataset_problems)

    expected_soultions = get_groundtruth(
        dataset_name=dataset,
        dataset=dataset_problems,
        dataset_hash=dataset_hash,
        cache_dir_prefix=cache_dir_prefix,
    )
    if n_early_stopping is None:
        dataset_solutions = run_codegen(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            dataset_dict=dataset_problems,
            greedy=greedy,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
        )

        eval_results = run_solutions_tests(
            dataset=dataset,
            dataset_problems=dataset_problems,
            dataset_solutions=dataset_solutions,
            expected_solutions=expected_soultions,
            parallel=parallel,
            base_only=base_only,
            test_details=test_details,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )

        results = prepare_evaluate_results(
            dataset=dataset,
            dataset_hash=dataset_hash,
            dataset_problems=dataset_problems,
            dataset_solutions=dataset_solutions,
            eval_results=eval_results,
            base_only=base_only,
            test_details=test_details,
        )
        results["early_stopped"] = False
    else:
        logger.info(f"Evaluating with {n_early_stopping=}")
        dataset_problems_early, dataset_problems_rest = split_problems(
            dataset_problems, n_early_stopping
        )

        dataset_solutions_early = run_codegen(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            dataset_dict=dataset_problems_early,
            greedy=greedy,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
        )

        eval_results_early = run_solutions_tests(
            dataset=dataset,
            dataset_solutions=dataset_solutions_early,
            dataset_problems=dataset_problems_early,
            expected_solutions=expected_soultions,
            parallel=parallel,
            base_only=base_only,
            test_details=test_details,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )

        results_early = prepare_evaluate_results(
            dataset=dataset,
            dataset_hash=dataset_hash,
            dataset_problems=dataset_problems_early,
            dataset_solutions=dataset_solutions_early,
            eval_results=eval_results_early,
            base_only=base_only,
            test_details=test_details,
        )

        if is_eligible_for_early_stopping(
            dataset_problems_early, dataset_solutions_early, eval_results_early
        ):
            logger.info("Very poor results, performing early stopping...")
            results = results_early
            results["early_stopped"] = True
        else:
            dataset_solutions_rest = run_codegen(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                dataset_dict=dataset_problems_rest,
                greedy=greedy,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
            )

            eval_results_rest = run_solutions_tests(
                dataset=dataset,
                dataset_problems=dataset_problems_rest,
                dataset_solutions=dataset_solutions_rest,
                expected_solutions=expected_soultions,
                parallel=parallel,
                base_only=base_only,
                test_details=test_details,
                min_time_limit=min_time_limit,
                gt_time_limit_factor=gt_time_limit_factor,
            )
            dataset_solutions = dataset_solutions_early | dataset_solutions_rest
            eval_results = eval_results_early | eval_results_rest
            results = prepare_evaluate_results(
                dataset=dataset,
                dataset_hash=dataset_hash,
                dataset_problems=dataset_problems,
                dataset_solutions=dataset_solutions,
                eval_results=eval_results,
                base_only=base_only,
                test_details=test_details,
            )
            results["early_stopped"] = False

    time_evalplus = time.perf_counter() - t_start
    results["time_evalplus"] = time_evalplus

    logger.info("Base tests:")
    for k, v in results["pass_at_k"]["base"].items():
        logger.info(f"{k}:\t{v:.3f}")
    if "plus" in results["pass_at_k"]:
        logger.info("Plus tests:")
        for k, v in results["pass_at_k"]["plus"].items():
            logger.info(f"{k}:\t{v:.3f}")
    logger.info(f"{time_evalplus=:.2f} seconds")
    return results


ALLOWED_METRICS_MBPP = [
    {"mbpp"},
    {"mbpp_plus"},
    {"mbpp", "mbpp_plus"},
]

ALLOWED_METRICS_HUMANEVAL = [
    {"humaneval"},
    {"humaneval_plus"},
    {"humaneval", "humaneval_plus"},
]


class EvalPlusEvaluator:
    def __init__(
        self,
        tokenizer,
        evaluator_metrics: dict[str, float],
        enable_thinking: Optional[bool],
        max_new_tokens: Optional[int],
        n_early_stopping: Optional[int],
        cache_dir_prefix: str,
    ):
        self.tokenizer = tokenizer
        self.evaluator_metrics = evaluator_metrics
        self.enable_thinking = enable_thinking
        evaluator_metrics_names = set(evaluator_metrics.keys())
        if evaluator_metrics_names in ALLOWED_METRICS_MBPP:
            self.dataset = "mbpp"
        elif evaluator_metrics_names in ALLOWED_METRICS_HUMANEVAL:
            self.dataset = "humaneval"
        else:
            raise ValueError(f"Unsupported set of metrics {evaluator_metrics_names}")

        evaluator_metrics_limits = list(evaluator_metrics.values())
        assert len(evaluator_metrics_limits) == 1 or len(evaluator_metrics_limits) == 2

        if (
            len(evaluator_metrics_limits) == 2
            and evaluator_metrics_limits[0] != evaluator_metrics_limits[1]
        ):
            msg = "Provided different limits for metrics, this is not supported"
            raise ValueError(msg)

        self.limit = evaluator_metrics_limits[0]
        self.greedy = True
        self.max_new_tokens = max_new_tokens
        self.n_early_stopping = n_early_stopping
        self.cache_dir_prefix = cache_dir_prefix
        self.last_results = None

    def get_last_results(self):
        return self.last_results

    def __call__(self, model: torch.nn.Module, device: torch.device):

        res_full = evaluate(
            model=model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            greedy=self.greedy,
            enable_thinking=self.enable_thinking,
            limit=self.limit,
            max_new_tokens=self.max_new_tokens,
            n_early_stopping=self.n_early_stopping,
            cache_dir_prefix=self.cache_dir_prefix,
        )
        res = {}
        if self.dataset == "mbpp":
            res["mbpp"] = float(res_full["pass_at_k"]["base"]["pass@1"])
            res["mbpp_plus"] = float(res_full["pass_at_k"]["plus"]["pass@1"])
        elif self.dataset == "humaneval":
            res["humaneval"] = float(res_full["pass_at_k"]["base"]["pass@1"])
            res["humaneval_plus"] = float(res_full["pass_at_k"]["plus"]["pass@1"])

        res["time_evalplus"] = res_full["time_evalplus"]
        self.last_results = res_full
        return res
