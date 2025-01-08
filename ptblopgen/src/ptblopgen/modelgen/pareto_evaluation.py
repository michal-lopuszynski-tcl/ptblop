import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any

import ptblop
import torch

from .. import builders, utils

logger = logging.getLogger(__name__)


@dataclass
class BPConfigProcsessingEvalEnvironment:
    run_id: str
    model: torch.nn.Module
    model_metadata: dict[str, Any]
    device: torch.device
    evaluator_fn: Any
    stop_path: pathlib.Path
    bp_config_db_path: pathlib.Path

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


def make_pareto_evaluated_path(pareto_path: pathlib.Path) -> pathlib.Path:
    return pareto_path.parent / (pareto_path.stem + "_evaluated.json")


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


def main_pareto_eval(
    config: dict[str, Any],
    pareto_path: pathlib.Path,
    min_metric: float,
) -> None:
    pareto_evaluated_path = make_pareto_evaluated_path(pareto_path)
    processing_env = BPConfigProcsessingEvalEnvironment(config)

    pareto_front_data = read_pareto_front_data(pareto_path)
    processed_bp_config_signatures = read_processed_bp_config_signatures(
        pareto_evaluated_path
    )
    for d in pareto_front_data:
        bpcs = utils.bp_config_signature_from_str(d["bp_config_signature"])
        if bpcs in processed_bp_config_signatures:
            logger.info(f"Skipping bp_config {bpcs}")
        else:
            logger.info(f"Processing bp_config {bpcs}")
            bp_config = d["bp_config"]
            res = process_bp_config(bp_config, processing_env)
            d["evaluation"] = res
            utils.update_db(pareto_evaluated_path, d)
            signature = utils.get_bp_config_signature(bp_config)
            processed_bp_config_signatures.add(signature)
