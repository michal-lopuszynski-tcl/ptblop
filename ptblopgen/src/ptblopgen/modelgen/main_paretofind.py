import pathlib
from dataclasses import dataclass
from typing import Any

import ptblop
import torch

from .. import builders, utils
from . import configurator
from . import estimator_helpers
from . import pareto_optimization


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
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model, evaluator_fn = builders.make_model_and_evaluator(
            config["model"], config["evaluator"], self.device
        )
        ptblop.apply_bp_config_in_place(model, {})
        self.model = model
        self.model_metadata = config["model"]
        self.evaluator_fn = evaluator_fn


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
    processing_env = BPConfigProcsessingFindEnvironment(config, bp_config_db_path)

    bp_config_unpruned = ptblop.get_unpruned_bp_config(processing_env.model)

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
