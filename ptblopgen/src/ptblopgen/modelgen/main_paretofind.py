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
    bp_config_db_path: pathlib.Path

    def __init__(self, config: dict[str, Any], bp_config_db_path: pathlib.Path):
        self.bp_config_db_path = bp_config_db_path
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
    bp_config_db_path: pathlib.Path
):
    config_sampler = configurator.SamplerConfig(**config["sampler"])
    config_pareto_optimization = configurator.ParetoOptimizationConfig(
        **config["pareto_optimization"]
    )
    pf_basename = pareto_optimization.PARETO_FRONT_FNAME_TEMPLATE % 0
    pareto_front_path = output_path / pareto_optimization.PARETO_FRONT_DIR / pf_basename

    quality_estimators_db_path = (
        output_path / pareto_optimization.QUALITY_ESTIMATOR_DB_FNAME
    )
    cost_estimators_db_path = output_path / pareto_optimization.COST_ESTIMATOR_DB_FNAME
    quality_estimator_report_path = (
        output_path / pareto_optimization.QUALITY_ESTIMATOR_REPORT_DIR
    )

    bp_config_unpruned = get_unpruned_bp_config(bp_config_db_path)

    processing_env = BPConfigProcsessingFindEnvironment(config, bp_config_db_path)

    quality_estimator_report_path.mkdir(exist_ok=False, parents=True)

    cost_estimator, _, cost_estimator_id = estimator_helpers.train_param_estimator(
        bp_config_db_path=processing_env.bp_config_db_path,
        cost_estimators_db_path=cost_estimators_db_path,
        data_iter=0,
        run_id=processing_env.run_id,
    )

    quality_estimator, quality_estimator_metrics, quality_estimator_id = (
        estimator_helpers.train_quality_estimator(
            bp_config_db_path=processing_env.bp_config_db_path,
            quality_estimator_report_path=quality_estimator_report_path,
            quality_estimators_db_path=quality_estimators_db_path,
            data_iter=0,
            quality_metric=config_sampler.quality_evaluator_metric,
            run_id=processing_env.run_id,
        )
    )

    pareto_optimization.find_pareto_front(
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
