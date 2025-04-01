import codecs
import collections
import logging
import random
import time
from typing import Any

import datasets
import lm_eval
import torch
import transformers

_SLEEP_SECONDS_ON_EXCEPTION = 30
_PPL_N_SAMPLES = 1000
_LOADER_SEED = 42


logger = logging.getLogger(__name__)


def _sync_gpus() -> None:
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)


def _map_tensors(
    obj: Any, device: torch.device | str | None = None, dtype: torch.dtype | None = None
) -> Any:
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        d = {k: _map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
        return d
    else:
        return obj


def _normalize_separator(
    separator: str, tokenizer: transformers.PreTrainedTokenizerBase
) -> str:

    allowed_separators = {"\n\n", " ", "", "eos"}

    # Hmm, ... brutal but it might work
    if separator not in allowed_separators:
        raise ValueError(f"{separator=} not in {allowed_separators=}")
    if separator == "eos":
        separator = tokenizer.eos_token
    return separator


def _escape_separator(separator: str) -> str:
    return codecs.escape_encode(separator.encode("utf-8"))[0].decode("utf-8")


def prepare_dataloader_v1(
    *,
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    separator: str,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: int = 128,
    varied_seqlen: bool = False,
    seed: int = 42,
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:

    separator = _normalize_separator(separator, tokenizer)

    logger.info(f"v1 dataloader - using sep={_escape_separator(separator)}")

    if not varied_seqlen and not nsamples:
        logger.warning(
            "varied_seqlen=False, but nsamples is not specified. This will lead to "
            "tokenization of the entire dataset, which will be slow."
        )

    msg = ", ".join(dataset.column_names)
    assert len(dataset.column_names) == 1, f"More than one column detected: {msg}"
    data_name = dataset.column_names[0]

    logger.info(f"v1 dataloader - using data column={data_name}")
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        # Create a new dataset where each example is a concatenation of multiple
        # examples of total length = max_seqlen
        data_list = ds[data_name]
        new_data_list: list[str] = []
        generator = torch.Generator()
        generator.manual_seed(seed)

        indices = list(range(len(data_list)))

        while (nsamples < 0 or len(new_data_list) < nsamples) and len(indices) > 0:
            start_idx = int(
                torch.randint(0, len(indices), (1,), generator=generator).item()
            )
            idx = start_idx
            tokens: list[str] = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                sep = "" if not tokens else separator
                tokens += tokenizer.tokenize(sep + item)
                idx += 1
            # logger.info(f"Used {idx-start_idx} examples")
            indices = indices[:start_idx] + indices[idx:]  # remove the used indices

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                new_data_list.append(tokenizer.convert_tokens_to_string(tokens))
        msg = f"v1 dataloader - created dataset of size {len(new_data_list)}"
        logger.info(msg)
        ds = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(
        data_batch: dict[str, torch.Tensor],
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    ds.set_transform(tokenize)
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(ds), generator=generator)[:nsamples].tolist()
    sampler = torch.utils.data.SubsetRandomSampler(indices)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return loader


def _remove_all_but_selected_columns(
    ds: datasets.Dataset,
    split_name: str,
    selected_columns: collections.abc.Iterable | collections.abc.Container,
) -> datasets.Dataset:
    cols_to_remove = [
        c for c in ds[split_name].column_names if c not in selected_columns
    ]
    if cols_to_remove:
        cols_to_remove_str = ", ".join(cols_to_remove)
        logger.info(f"Removing columns: {cols_to_remove_str}")
        ds = ds.remove_columns(cols_to_remove)
    return ds


def get_dataset(dataset_and_split_name: str) -> datasets.Dataset:
    DS_PROPERTIES: dict[str, dict[str, Any]] = {
        "wikitext2": {"path": "wikitext", "config_name": "wikitext-2-raw-v1"},
        "alpaca": {
            "path": "tatsu-lab/alpaca",
            "data_column": "text",
        },
    }
    ds_available = set(DS_PROPERTIES.keys())
    dataset_name, split_name = dataset_and_split_name.split(".")
    if dataset_name not in ds_available:
        raise ValueError(f"Unkown dataset {dataset_name}, available are {ds_available}")

    properties = DS_PROPERTIES[dataset_name]

    while True:
        try:
            ds = datasets.load_dataset(
                properties["path"],
                name=properties.get("config_name"),
                data_files=properties.get("data_files"),
            )
            break
        except Exception as e:
            logger.warning(f"Exception {e} during creating wikitext")
            time.sleep(_SLEEP_SECONDS_ON_EXCEPTION)

    if dataset_name == "alpaca":
        if split_name == "full":
            split_name = "train"
        else:
            # Alpaca does not have valid/test, so create custom valid/test 10% splits
            ds = ds["train"].train_test_split(test_size=0.2, seed=42)
            temp_ds = ds.pop("test")
            temp_ds = temp_ds.train_test_split(test_size=0.5, seed=42)
            ds["test"] = temp_ds["train"]
            ds["validation"] = temp_ds["test"]

    if "data_column" in properties:
        ds = _remove_all_but_selected_columns(
            ds, split_name=split_name, selected_columns=properties["data_column"]
        )
    res = ds[split_name]
    msg = ", ".join(res.column_names)
    assert len(res.column_names) == 1, f"More than one column detected: {msg}"

    return res


def calc_lm_eval_metrics(
    model: torch.nn.Module,
    tasks: list[str],
    tokenizer: transformers.PreTrainedTokenizerBase,
    device: torch.device,
) -> tuple[dict[str, Any], str]:

    lm_eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=model, tokenizer=tokenizer, device=device
    )
    if isinstance(tasks, dict):
        results = {}
        for task, limit in tasks.items():
            results_task = lm_eval.evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=[task],
                batch_size="auto",
                device=device,
                limit=limit,
                confirm_run_unsafe_code=True,
            )
            results_task["config"]["device"] = str(results["config"]["device"])
            results_task["config"]["model_dtype"] = str(results["config"]["device"])
            results[task] = results_task
            return results
    # TODO Remove this
    # elif isinstance(tasks, list):
    #     results = lm_eval.evaluator.simple_evaluate(
    #         model=lm_eval_model,
    #         tasks=tasks,
    #         batch_size="auto",
    #         device=device,
    #         confirm_run_unsafe_code=True,
    #     )

    #     # Make results JSON-serializeable
    #     results["config"]["device"] = str(results["config"]["device"])
    #     results["config"]["model_dtype"] = str(results["config"]["device"])
    #     return results
    else:
        raise ValueError(f"Unknown {type(task)=}")


def calc_perplexity(
    model: torch.nn.Module,
    testloader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    pad_token_id: int,
) -> float:
    _sync_gpus()

    with torch.no_grad():
        model.eval()

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)

        nlls = []

        logger.info("Perplexity evaluation started")
        for batch in testloader:
            batch = _map_tensors(batch, device=device)

            logits = model(**batch).logits

            logits = logits[:, :-1, :]
            shift_labels = batch["input_ids"][:, 1:]

            nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

            mask = shift_labels != loss_fn.ignore_index
            nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
            nlls.append(nll_means)

        nlls_tensor = torch.cat(nlls)
        perplexity = torch.exp(nlls_tensor.mean())
        res = perplexity.item()

    _sync_gpus()

    return res


def make_dataloader_perplexity(
    tokenizer: transformers.PreTrainedTokenizer,
) -> tuple[torch.utils.data.DataLoader, int]:

    perplexity_data_name = "wikitext2.test"
    perplexity_data_separator = (
        "\n\n"  # Normal pythonic string, no extra escaping needed
    )
    perplexity_data_max_length = 1024
    perplexity_data_batch_size = 1

    perplexity_ds = get_dataset(perplexity_data_name)
    perplexity_n = len(perplexity_ds)
    msg = f"Created perplexity dataset {perplexity_data_name}, "
    msg += f"{perplexity_n} examples"
    logger.info(msg)

    perplexity_dl = prepare_dataloader_v1(
        dataset=perplexity_ds,
        tokenizer=tokenizer,
        max_seqlen=perplexity_data_max_length,
        batch_size=perplexity_data_batch_size,
        separator=perplexity_data_separator,
        nsamples=_PPL_N_SAMPLES,
        varied_seqlen=False,
        seed=_LOADER_SEED,
    )

    return perplexity_dl, perplexity_n


class LMEvalWithPPLEvaluator:
    def __init__(self, tokenizer, evaluator_metrics):
        if "ppl" in evaluator_metrics:
            self.ppl_dl, _ = make_dataloader_perplexity(tokenizer)
        else:
            self.ppl_dl = None
        self.tokenizer = tokenizer
        self.lm_eval_tasks = [em for em in evaluator_metrics if em != "ppl"]

    def __call__(self, model: torch.nn.Module, device: torch.device):

        model.eval()
        if self.ppl_dl is not None:
            t1 = time.perf_counter()
            perplexity = calc_perplexity(
                model=model,
                testloader=self.ppl_dl,
                device=device,
                pad_token_id=model.config.pad_token_id,
            )
            t2 = time.perf_counter()
            time_perplex_eval = t2 - t1
            res_ppl = {
                "ppl": perplexity,
                "time_ppl": time_perplex_eval,
            }
        else:
            res_ppl = {}

        if self.lm_eval_tasks:
            t1 = time.perf_counter()
            while True:
                try:
                    res_dict = calc_lm_eval_metrics(
                        model=model,
                        tasks=self.lm_eval_tasks,
                        tokenizer=self.tokenizer,
                        device=device,
                    )
                    break
                except Exception as e:
                    logger.warning(f"Exception {e} during lm_eval")
                    time.sleep(_SLEEP_SECONDS_ON_EXCEPTION)
            t2 = time.perf_counter()

            res_lm_eval = {}
            for task in self.lm_eval_tasks:
                # TODO Remove this
                # if "results" in res_dict:
                #     res_dict_task = res_dict["results"][task]
                # else:
                #     # This is for the case when task is dictionary task to limit
                res_dict_task = res_dict[task]["results"][task]
                if "acc,none" in res_dict_task:
                    res_lm_eval[task] = res_dict_task["acc,none"]
                    res_lm_eval[f"{task}_stderr"] = res_dict_task["acc_stderr,none"]
                elif "pass_at_1,none" in res_dict_task:
                    res_lm_eval[f"{task}"] = res_dict_task["pass_at_1,none"]
                    res_lm_eval[f"{task}_stderr"] = res_dict_task[
                        "pass_at_1_stderr,none"
                    ]
                else:
                    ValueError("No known_metric found - acc, pass_at_1")

            res_lm_eval["time_lm_eval"] = t2 - t1
        else:
            res_lm_eval = {}
        res = res_ppl | res_lm_eval
        return res


class MockLMEvalWithPPLEvaluator:
    def __init__(self, tokenizer, evaluator_metrics, evaluator_seed):
        if "ppl" in evaluator_metrics:
            self.ppl_dl = True
        self.tokenizer = tokenizer
        self.lm_eval_tasks = [em for em in evaluator_metrics if em != "ppl"]
        self.rng = random.Random(evaluator_seed)

    def __call__(self, model: torch.nn.Module, device: torch.device):
        if self.ppl_dl:
            t1 = time.perf_counter()
            perplexity = 1000.0 * self.rng.random()
            t2 = time.perf_counter()
            time_perplex_eval = t2 - t1
            res_ppl = {
                "ppl": perplexity,
                "time_ppl": time_perplex_eval,
            }
        else:
            res_ppl = {}

        if self.lm_eval_tasks:
            t1 = time.perf_counter()

            res_lm_eval = {}
            for task in self.lm_eval_tasks:
                task_score = self.rng.random()
                res_lm_eval[f"{task}"] = task_score
                # Add max 20% relative error
                res_lm_eval[f"{task}_acc_stderr"] = task_score * 0.2 * self.rng.random()
            t2 = time.perf_counter()

            res_lm_eval["time_lm_eval"] = t2 - t1
        else:
            res_lm_eval = {}
        res = res_ppl | res_lm_eval
        return res


def make_evaluator(evaluator_config, tokenizer):
    evaluator_name = evaluator_config["evaluator_name"]
    if evaluator_name == "lm_eval_with_ppl":
        return LMEvalWithPPLEvaluator(
            tokenizer,
            evaluator_config["evaluator_metrics"],
            evaluator_config["evaluator_limit"],
        )
    elif evaluator_name == "mock_lm_eval_with_ppl":
        return MockLMEvalWithPPLEvaluator(
            tokenizer,
            evaluator_config["evaluator_metrics"],
            evaluator_seed=evaluator_config["evaluator_seed"],
        )
    else:
        raise ValueError(f"Unsupported evaluator - {evaluator_name=}")
