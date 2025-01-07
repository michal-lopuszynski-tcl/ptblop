import json
import logging
import pathlib
from typing import Any

from .. import utils
from . import configurator, modelgen


logger = logging.getLogger(__name__)


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


def main_pareto_eval(
    config: dict[str, Any],
    output_path: pathlib.Path,
    pareto_path: pathlib.Path,
    min_metric: float,
) -> None:
    run_id = "dummy"
    # config_sampler = configurator.SamplerConfig(**config["sampler"])
    # config_pareto_optimization = configurator.ParetoOptimizationConfig(
    #     **config["pareto_optimization"]
    # )
    pareto_evaluated_path = make_pareto_evaluated_path(pareto_path)

    processing_env = modelgen.make_bp_config_processing_env(config, output_path, run_id)

    pareto_front_data = read_pareto_front_data(pareto_path)
    processed_bp_config_signatures = read_processed_bp_config_signatures(
        pareto_evaluated_path
    )
    for d in pareto_front_data:
        bpcs = utils.bp_config_signature_from_str(d["bp_config_signature"])
        if bpcs in processed_bp_config_signatures:
            logger.info(f"Skipping bp_config {bpcs}")
        else:
            bp_config = d["bp_config"]
            res, _ = modelgen.process_bp_config(
                bp_config=bp_config,
                bp_config_id="dummy",
                bp_config_type="dummy",
                bp_config_score=1.0,
                data_iter=1,
                processing_env=processing_env,
                processed_bp_config_signatures=processed_bp_config_signatures,
            )
            d["evaluation"] = res["evaluation"]
            utils.update_db(pareto_evaluated_path, d)
