import json
import pathlib
from dataclasses import dataclass
from typing import Any

import torch

from .. import utils
from . import configurator, estimator_helpers, pareto_optimization


@dataclass
class BPConfigProcsessingFindEnvironment:
    run_id: str
    model: torch.nn.Module
    model_metadata: dict[str, Any]
    device: torch.device
    evaluator_fn: Any
    stop_path: pathlib.Path
    bp_config_db_paths: list[pathlib.Path]

    def __init__(self, config: dict[str, Any], bp_config_db_paths: pathlib.Path):
        self.bp_config_db_paths = bp_config_db_paths
        self.run_id = utils.make_runid()
        self.model_metadata = config["model"]


def get_unpruned_bp_config(bp_config_path: pathlib.Path):
    with open(bp_config_path, "rt") as f:
        d = json.loads(f.readline())
    bp_config = d["bp_config"]

    for k1 in bp_config:
        for k2 in k1:
            bp_config[k1][k2] = True
    return bp_config


def main_paretofind(
    *,
    config: dict[str, Any],
    output_path: pathlib.Path,
    bp_config_db_paths: pathlib.Path,
):
    config_sampler = configurator.SamplerConfig(**config["sampler"])

    pareto_optimizer = config["pareto_optimization"]["optimizer"]

    pf_basename = pareto_optimization.PARETO_FRONT_FNAME_TEMPLATE % 0
    pareto_front_path = output_path / pareto_optimization.PARETO_FRONT_DIR / pf_basename

    quality_estimators_db_path = (
        output_path / pareto_optimization.QUALITY_ESTIMATOR_DB_FNAME
    )
    cost_estimators_db_path = output_path / pareto_optimization.COST_ESTIMATOR_DB_FNAME
    quality_estimator_report_path = (
        output_path / pareto_optimization.QUALITY_ESTIMATOR_REPORT_DIR
    )

    bp_config_unpruned = get_unpruned_bp_config(bp_config_db_paths[0])

    processing_env = BPConfigProcsessingFindEnvironment(config, bp_config_db_paths)

    quality_estimator_report_path.mkdir(exist_ok=False, parents=True)

    cost_estimator, _, cost_estimator_id = estimator_helpers.train_param_estimator(
        bp_config_db_paths=processing_env.bp_config_db_paths,
        cost_estimators_db_path=cost_estimators_db_path,
        data_iter=0,
        run_id=processing_env.run_id,
        full_block_mode=config_sampler.full_block_mode,
    )

    quality_estimator, quality_estimator_metrics, quality_estimator_id = (
        estimator_helpers.train_quality_estimator(
            quality_estimator_config=config["quality_estimator"],
            bp_config_db_paths=processing_env.bp_config_db_paths,
            quality_estimator_report_path=quality_estimator_report_path,
            quality_estimators_db_path=quality_estimators_db_path,
            data_iter=0,
            quality_metric=config_sampler.quality_evaluator_metric,
            run_id=processing_env.run_id,
            full_block_mode=config_sampler.full_block_mode,
        )
    )

    full_block_mode = config_sampler.full_block_mode
    if pareto_optimizer == "beam":
        config_pareto_optimization = configurator.BeamParetoOptimizationConfig(
            **config["pareto_optimization"]
        )
        if full_block_mode:
            pareto_optimization.find_pareto_front_beam_full_block(
                run_id=processing_env.run_id,
                model_metadata=processing_env.model_metadata,
                quality_estimator=quality_estimator,
                quality_estimator_id=quality_estimator_id,
                quality_metric_name=config_sampler.quality_evaluator_metric,
                cost_estimator=cost_estimator,
                cost_estimator_id=cost_estimator_id,
                n_features=quality_estimator_metrics["n_features_trn"],
                bp_config_unpruned=bp_config_unpruned,
                pareto_path=pareto_front_path,
                config_pareto_optimization=config_pareto_optimization,
            )
        else:
            pareto_optimization.find_pareto_front_beam_non_full_block(
                run_id=processing_env.run_id,
                model_metadata=processing_env.model_metadata,
                quality_estimator=quality_estimator,
                quality_estimator_id=quality_estimator_id,
                quality_metric_name=config_sampler.quality_evaluator_metric,
                cost_estimator=cost_estimator,
                cost_estimator_id=cost_estimator_id,
                n_features=quality_estimator_metrics["n_features_trn"],
                bp_config_unpruned=bp_config_unpruned,
                pareto_path=pareto_front_path,
                config_pareto_optimization=config_pareto_optimization,
            )
    elif pareto_optimizer == "pymoo":
        config_pareto_optimization = configurator.PymooParetoOptimizationConfig(
            **config["pareto_optimization"]
        )
        pareto_optimization.find_pareto_front_pymoo(
            run_id=processing_env.run_id,
            model_metadata=processing_env.model_metadata,
            quality_estimator=quality_estimator,
            quality_estimator_id=quality_estimator_id,
            quality_metric_name=config_sampler.quality_evaluator_metric,
            cost_estimator=cost_estimator,
            cost_estimator_id=cost_estimator_id,
            n_features=quality_estimator_metrics["n_features_trn"],
            bp_config_unpruned=bp_config_unpruned,
            pareto_path=pareto_front_path,
            config_pareto_optimization=config_pareto_optimization,
            full_block_mode=full_block_mode,
        )
    else:
        raise ValueError(f"Unknown Pareto optimizer {pareto_optimizer}")
