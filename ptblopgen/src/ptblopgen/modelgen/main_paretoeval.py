import json
import logging
import pathlib
import random
import shutil
from dataclasses import dataclass
from typing import Any, Optional

import ptblop
import torch

from .. import builders, utils

logger = logging.getLogger(__name__)


@dataclass
class BPConfigProcsessingEvalEnvironment:
    run_id: str
    device: torch.device
    model: torch.nn.Module
    evaluator_fn: Any

    def __init__(self, config: dict[str, Any]):
        self.run_id = utils.make_runid()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model, evaluator_fn = builders.make_model_and_evaluator(
            config["model"], config["evaluator"], self.device
        )
        ptblop.apply_bp_config_in_place(model, {})
        self.model = model
        self.evaluator_fn = evaluator_fn


def make_pareto_evaluated_paths(
    pareto_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    p1 = pareto_path.parent / (pareto_path.stem + "_evaluated.json")
    p2 = pareto_path.parent / (pareto_path.stem + "_evaluated.json.bak")
    return p1, p2


def read_processed_bp_config_signatures(pareto_path: pathlib.Path) -> set[int]:
    proessed_bp_config_signatures = set()
    if pareto_path.exists():
        with open(pareto_path, "rt") as f:
            for line in f:
                d = json.loads(line)
                bpcs = utils.bp_config_signature_from_str(d["bp_config_signature"])
                proessed_bp_config_signatures.add(bpcs)
    return proessed_bp_config_signatures


def read_pareto_front_data(pareto_path: pathlib.Path):
    with open(pareto_path, "rt") as f:
        pf_data_raw = [json.loads(line) for line in f]
    return pf_data_raw


def process_bp_config(
    bp_config,
    processing_env: BPConfigProcsessingEvalEnvironment,
):
    res = {
        "run_id": processing_env.run_id,
    }
    ptblop.apply_bp_config_in_place(
        processing_env.model, bp_config, set_unused_layers_to_none=False
    )
    res["mparams"] = ptblop.get_num_active_params(processing_env.model) / 1.0e6
    res_evaluation = processing_env.evaluator_fn(
        processing_env.model, processing_env.device
    )
    res |= res_evaluation
    res["timestamp"] = utils.get_timestamp()
    device_str = str(processing_env.device)
    if "cuda" in device_str:
        device_str += " @ " + torch.cuda.get_device_name(processing_env.device)
    res["device"] = device_str
    v_ptblop, v_ptblopgen = utils.get_versions()
    res["ptblop_version"] = v_ptblop
    res["ptblopgen_version"] = v_ptblopgen
    return res


def update_pareto_front(
    db_path: pathlib.Path,
    db_path_bak: pathlib.Path,
    db_entry: dict[str, Any],
) -> None:
    if not db_path.exists():
        with open(db_path, "wt") as f:
            f.write(json.dumps(db_entry) + "\n")
    else:
        # Read data
        with open(db_path, "rt") as f:
            pareto_data = [json.loads(line) for line in f]

        # Append new record
        pareto_data.append(db_entry)

        # Sort
        pareto_data.sort(key=lambda d: -d["mparams_pred"])

        # Copy old data to backup, in case job gets killed or exception occurs
        shutil.copy2(db_path, db_path_bak)

        # Save updated data

        with open(db_path, "wt") as f:
            for di in pareto_data:
                f.write(json.dumps(di) + "\n")


def filter_processed(pareto_front_data, processed_bp_config_signatures):
    res = []
    for d in pareto_front_data:
        bpcs = utils.bp_config_signature_from_str(d["bp_config_signature"])
        if bpcs not in processed_bp_config_signatures:
            res.append(d)
    return res


def filter_by_min_metric(pareto_front_data, config, min_metric):
    res = []
    metric_key = config["sampler"]["quality_evaluator_metric"] + "_pred"
    for d in pareto_front_data:
        if d[metric_key] >= min_metric:
            res.append(d)
    return res


def main_paretoeval(
    *,
    config: dict[str, Any],
    pareto_path: pathlib.Path,
    min_metric: Optional[float],
    shuffle: bool,
) -> None:

    pareto_front_data = read_pareto_front_data(pareto_path)
    n_tot = len(pareto_front_data)
    logger.info(f"Read {n_tot=} configurations from {pareto_path}")

    pareto_evaluated_path, pareto_evaluated_path_bak = make_pareto_evaluated_paths(
        pareto_path
    )
    processed_bp_config_signatures = read_processed_bp_config_signatures(
        pareto_evaluated_path
    )
    n_evaluated = len(processed_bp_config_signatures)
    logger.info(f"Read {n_evaluated=} from {pareto_evaluated_path}")

    processing_env = BPConfigProcsessingEvalEnvironment(config)
    pareto_front_data = filter_processed(
        pareto_front_data, processed_bp_config_signatures
    )
    n_tot = len(pareto_front_data)
    logger.info(f"Filtering processed - {n_tot=} left")

    if min_metric is not None:
        pareto_front_data = filter_by_min_metric(pareto_front_data, config, min_metric)
        n_tot = len(pareto_front_data)
        logger.info(f"Filtering by {min_metric=} - {n_tot=} left")
    else:
        logger.info(f"Filtering by {min_metric=} - skipped")

    if shuffle:
        logger.info("Shuffling pareto front data before processing")
        random.shuffle(pareto_front_data)
    else:
        logger.info("Skipping shuffling pareto front data")

    for i, d in enumerate(pareto_front_data, start=1):
        bpcs = utils.bp_config_signature_from_str(d["bp_config_signature"])
        if bpcs in processed_bp_config_signatures:
            logger.info(f"Skipping bp_config {bpcs}")
        else:
            logger.info(f"Processing bp_config {i} out of {n_tot} - {bpcs=}")
            bp_config = d["bp_config"]
            res = process_bp_config(bp_config, processing_env)
            d["evaluation"] = res
            update_pareto_front(pareto_evaluated_path, pareto_evaluated_path_bak, d)
            signature = utils.get_bp_config_signature(bp_config)
            processed_bp_config_signatures.add(signature)
    pareto_evaluated_path_bak.unlink(missing_ok=True)
