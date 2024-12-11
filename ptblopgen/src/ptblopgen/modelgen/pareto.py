import argparse
import json
import logging
import sys

import numpy as np
import pymoo.algorithms.moo.nsga2
import pymoo.core.problem
import pymoo.operators.crossover.pntx
import pymoo.operators.mutation.bitflip
import pymoo.operators.sampling.rnd
import pymoo.optimize
import pymoo.termination
import sklearn.linear_model

from . import estimator_helpers

TARGET_COLUMN = "arc_challenge_acc"

DB_FNAME = "bp_configs.json"


logger = logging.getLogger(__name__)


# def setup_logging():
#     logging.basicConfig(
#         level=logging.WARNING,
#         format="%(asctime)s.%(msecs)03d500: %(levelname).1s %(name)s.py:%(lineno)d] %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )
#     # Here you put modules where you want more verbose logging

#     for module_name in [__name__, "blockprunekit"]:
#         logging.getLogger(module_name).setLevel(logging.INFO)


# def parse_args() -> argparse.Namespace:
#     # Try to parse --version

#     arg_parser = argparse.ArgumentParser()
#     arg_parser.add_argument("--version", action="store_true")
#     arg_parser.add_argument("--output-path", type=pathlib.Path)
#     # arg_parser.add_argument("--config", type=pathlib.Path)

#     args = arg_parser.parse_args()

#     # If no --version, run parsing of trainign/decomposition arguments

#     if not args.version:
#         arg_parser = argparse.ArgumentParser()
#         arg_parser.add_argument(
#             "--output-path",
#             type=pathlib.Path,
#             required=True,
#         )
#         # arg_parser.add_argument(
#         #     "--config",
#         #     type=pathlib.Path,
#         #     required=True,
#         # )
#         args = arg_parser.parse_args()
#         args.version = False

#     return args


# def quality_regressor_fit(
#     data_trn, data_val, target_column, regressor_dir, reg_kwargs=None
# ):

#     X_trn = estimator_helpers.get_quality_features([d["bp_config"] for d in data_trn])
#     y_trn = estimator_helpers.get_target(data_trn, target_column)
#     logger.info(f"{X_trn.shape=} {X_trn.dtype=} {y_trn.shape=} {y_trn.dtype=}")

#     X_val = estimator_helpers.get_quality_features([d["bp_config"] for d in data_val])
#     y_val = estimator_helpers.get_target(data_val, target_column)
#     logger.info(f"{X_val.shape=} {X_val.dtype=} {y_val.shape=} {y_val.dtype=}")

#     n_examples_trn, n_features_trn = X_trn.shape
#     n_examples_val, n_features_val = X_val.shape

#     # GBM Regressor
#     reg_type = "QuantileGradientBoostingBoundsRegressor"
#     reg_kwargs = dict(
#         learning_rate=0.03,
#         n_estimators=200,
#         max_depth=6,
#         min_samples_leaf=9,
#         min_samples_split=9,
#     )
#     reg = blockprunekit.regressors.QuantileGradientBoostingBoundsRegressor(**reg_kwargs)
#     reg.fit(X_trn, y_trn)

#     # Linear regressor
#     # reg_type = "QuantileLinearRegressor"
#     # reg_kwargs = dict(fit_intercept=True, alpha=0.01)
#     # reg = blockprunekit.regressors.QuantileLinearRegressor(**reg_kwargs)

#     # Random forest regressor
#     # reg_type = "EnsembleRandomForestBoundsRegressor"
#     # if reg_kwargs is None:
#     #     reg_kwargs = dict(n_regressors=20, n_estimators=300)
#     # # if reg_kwargs is None:
#     # #     reg_kwargs = dict(
#     # #         n_regressors=20, n_estimators=300, max_features=0.8, min_samples_leaf=2
#     # #     )
#     # reg = blockprunekit.regressors.EnsembleRandomForestBoundsRegressor(**reg_kwargs)
#     # reg.fit(X_trn, y_trn)

#     # Extra trees regressor
#     # reg_type = "EnsembleExtraTreesBoundsRegressor"
#     # if reg_kwargs is None:
#     #     reg_kwargs = dict(
#     #         n_regressors=20, n_estimators=300, max_features=0.8, bootstrap=True
#     #     )
#     # reg = blockprunekit.regressors.EnsembleExtraTreesBoundsRegressor(**reg_kwargs)
#     # reg.fit(X_trn, y_trn)

#     # # Experimental Ensemble bradeint boosting regressor
#     # reg_type = "EnsembleGradientBoostingBoundsRegressor"
#     # reg_kwargs = dict(
#     #     n_regressors=20,
#     #     subsample=0.1,
#     #     learning_rate=0.03,
#     #     n_estimators=200,
#     #     max_depth=6,
#     #     min_samples_leaf=9,
#     #     min_samples_split=9,
#     # )
#     # reg = blockprunekit.regressors.EnsembleGradientBoostingBoundsRegressor(**reg_kwargs)
#     # reg.fit(X_trn, y_trn)

#     # Saving Regressor - not all regressors support that yet
#     # regressor_path = regressor_dir / "quality_regressor.json.gz"
#     # blockprunekit.regressors.save_regressor(regressor_path, reg)

#     regressor_plot_fname = regressor_dir / "quality_regressor_stats.png"
#     reg_metrics = estimator_helpers.evaluate_bounds_estimator(
#         bounds_regressor=reg,
#         X_trn=X_trn,
#         y_trn=y_trn,
#         X_val=X_val,
#         y_val=y_val,
#         plot_fname=regressor_plot_fname,
#     )
#     logger.info(f"{reg_metrics=}")

#     # reg1 = blockprunekit.regressors.load_regressor("quality_regressor.json.gz")
#     # reg_metrics1 = estimator_helpers.evaluate_bounds_regressor(
#     #     bounds_regressor=reg1, X_trn=X_trn, y_trn=y_trn, X_val=X_val, y_val=y_val,plot_fname="tmp.png",
#     # )
#     # logger.info(f"{reg_metrics1=}")

#     reg_data = {
#         "n_examples_trn": n_examples_trn,
#         "n_features_trn": n_features_trn,
#         "n_examples_val": n_examples_val,
#         "n_features_val": n_features_val,
#         "regressor_type": reg_type,
#         "regressor_kwargs": reg_kwargs,
#         "regressor_metrics": reg_metrics,
#         "blockprunekit_version": blockprunekit.__version__,
#     }
#     logger.info(f"{reg_data=}")
#     regressor_stats_fname = regressor_dir / "quality_regressor_stats.json"
#     with open(regressor_stats_fname, "wt") as f:
#         json.dump(reg_data, f)
#     return reg, reg_data


# def param_regressor_fit(data_trn, data_val):
#     X_trn = estimator_helpers.get_quality_features([d["bp_config"] for d in data_trn])
#     y_trn = estimator_helpers.get_target(data_trn, "mparams")
#     logger.info(f"{X_trn.shape=} {X_trn.dtype=} {y_trn.shape=} {y_trn.dtype=}")
#     X_val = estimator_helpers.get_quality_features([d["bp_config"] for d in data_val])
#     y_val = estimator_helpers.get_target(data_val, "mparams")
#     logger.info(f"{X_val.shape=} {X_val.dtype=} {y_val.shape=} {y_val.dtype=}")

#     n_examples_trn, n_features_trn = X_trn.shape
#     n_examples_val, n_features_val = X_val.shape

#     reg_type = "LinearRegression"
#     reg_kwargs = dict(fit_intercept=True)
#     reg = sklearn.linear_model.LinearRegression(**reg_kwargs)
#     reg.fit(X_trn, y_trn)

#     reg_metrics = estimator_helpers.evaluate_estimator(
#         bounds_regressor=reg,
#         X_trn=X_trn,
#         y_trn=y_trn,
#         X_val=X_val,
#         y_val=y_val,
#     )

#     reg_data = {
#         "n_examples_trn": n_examples_trn,
#         "n_features_trn": n_features_trn,
#         "n_examples_val": n_examples_val,
#         "n_features_val": n_features_val,
#         "regressor_type": reg_type,
#         "regressor_kwargs": reg_kwargs,
#         "regressor_metrics": reg_metrics,
#         "blockprunekit_version": blockprunekit.__version__,
#     }
#     logger.info(f"{reg_data=}")
#     return reg, reg_data


def build_predict_quality(reg_quality):
    def __predict_quality(x):
        quality, _, _ = reg_quality.predict_with_bounds(x)
        return -quality

    return __predict_quality


def build_predict_quality_min(reg_quality):
    def __predict_quality(x):
        _, quality, _ = reg_quality.predict_with_bounds(x)
        return -quality

    return __predict_quality


def build_predict_cost(reg_param):
    def __predict_param(x):
        param = reg_param.predict(x)
        return param

    return __predict_param


class BinaryProblem(pymoo.core.problem.Problem):
    def __init__(self, f1, f2, n_var):
        super().__init__(
            n_var=n_var,  # number of variables
            n_obj=2,  # number of objectives
            n_ieq_constr=0,  # number of constraints
            xl=0,  # lower bound
            xu=1,  # upper bound
            vtype=bool,  # binary type
        )
        self.f1 = f1
        self.f2 = f2

    def _evaluate(self, x, out, *args, **kwargs):
        # x has shape (population_size, 56)
        f1 = self.f1(x)  # returns array of shape (population_size,)
        f2 = self.f2(x)  # returns array of shape (population_size,)

        # Stack the objectives into shape (population_size, 2)
        out["F"] = np.column_stack([f1, f2])


def get_bp_config_from_features(sample_bp_config, features):
    assert len(features) == 2 * len(sample_bp_config)
    res = {}

    for i, k in enumerate(sample_bp_config):
        use_attention = float(features[2 * i]) > 0.5
        use_mlp = float(features[2 * i + 1]) > 0.5
        res[k] = {"use_attention": use_attention, "use_mlp": use_mlp}
    return res


def get_bp_config_stats(bp_config):
    n = len(bp_config)
    n_attention = sum(int(l["use_attention"]) for l in bp_config.values())
    n_mlp = sum(int(l["use_mlp"]) for l in bp_config.values())
    return n, n_attention, n_mlp


def main_tune_rf(args: argparse.Namespace):
    pareto_dir = args.output_path / "pareto"
    if pareto_dir.exists():
        print(f"Output already exists, please delete it `rm -rf {pareto_dir}`")
        sys.exit(1)
    pareto_dir.mkdir(exist_ok=True, parents=True)

    bp_config_db_path = args.output_path / DB_FNAME
    data_trn, data_val = estimator_helpers.read_data(bp_config_db_path)
    reg_kwargs_common = dict(n_regressors=20, n_estimators=300)

    min_samples_leaf_values = [1, 2, 4, 10]
    max_features_values = [0.1, 0.3, 0.5, 0.8, 1.0]

    reg_kwargs_values = []

    for min_samples_leaf in min_samples_leaf_values:
        for max_features in max_features_values:
            reg_kwargs = reg_kwargs_common | {
                "max_features": max_features,
                "min_samples_leaf": min_samples_leaf,
            }
            reg_kwargs_values.append(reg_kwargs)
    import random

    r = random.Random(42)
    r.shuffle(reg_kwargs_values)

    for i, reg_kwargs in enumerate(reg_kwargs_values, start=1):
        regressor_dir = pareto_dir / f"{i:03d}"
        regressor_dir.mkdir(parents=True, exist_ok=True)
        reg_quality, reg_quality_data = quality_regressor_fit(
            data_trn,
            data_val,
            target_column=TARGET_COLUMN,
            regressor_dir=regressor_dir,
            reg_kwargs=reg_kwargs,
        )


def find_pareto_front(
    *,
    quality_estimator,
    cost_estimator,
    bp_config_unpruned,
    n_features,
    pareto_path,
):
    # pareto_dir = args.output_path / "pareto"
    # if pareto_dir.exists():
    #     print(f"Output already exists, please delete it `rm -rf {pareto_dir}`")
    #     sys.exit(1)
    # pareto_dir.mkdir(exist_ok=True, parents=True)

    # bp_config_db_path = args.output_path / DB_FNAME
    # data_trn, data_val = estimator_helpers.read_data(
    #     bp_config_db_path
    # )
    # reg_quality, reg_quality_data = quality_regressor_fit(
    #     data_trn, data_val, target_column=TARGET_COLUMN, regressor_dir=pareto_dir
    # )
    # f_quality = build_predict_quality_min(reg_quality)
    f_quality = build_predict_quality(quality_estimator)
    # reg_param, reg_param_data = param_regressor_fit(data_trn, data_val)
    f_cost = build_predict_cost(cost_estimator)
    # n_features = quality_estimator_metrics["n_features_trn"]

    problem = BinaryProblem(f1=f_quality, f2=f_cost, n_var=n_features)

    algorithm = pymoo.algorithms.moo.nsga2.NSGA2(
        pop_size=2000,
        sampling=pymoo.operators.sampling.rnd.BinaryRandomSampling(),
        crossover=pymoo.operators.crossover.pntx.TwoPointCrossover(),
        mutation=pymoo.operators.mutation.bitflip.BitflipMutation(prob=1 / n_features),
        eliminate_duplicates=True,
    )

    termination = pymoo.termination.get_termination("n_gen", 500)

    res = pymoo.optimize.minimize(problem, algorithm, termination, seed=1, verbose=True)

    pareto_front = res.F

    # np.savetxt(pareto_dir / "pareto_front.csv", pareto_front, delimiter=",")
    # np.savetxt(pareto_dir / "pareto_solutions.csv", res.X, delimiter=",", fmt="%d")
    m, _ = pareto_front.shape

    pareto_data = []
    for i in range(m):
        features = res.X[i, :]
        # print(features)
        bp_config = get_bp_config_from_features(bp_config_unpruned, features)
        n, n_attention, n_mlp = get_bp_config_stats(bp_config)
        q, qmin, qmax = quality_estimator.predict_with_bounds([features])
        params = cost_estimator.predict([features])
        d = {
            "mparams_pred": params.item(),
            "n": n,
            "n_attention": n_attention,
            "n_mlp": n_mlp,
            f"{TARGET_COLUMN}_pred": q.item(),
            f"{TARGET_COLUMN}_pred_min": qmin.item(),
            f"{TARGET_COLUMN}_pred_max": qmax.item(),
            "bp_config": bp_config,
        }
        pareto_data.append(d)
    pareto_data = sorted(pareto_data, key=lambda d: -d[f"mparams_pred"])
    pareto_path.parent.mkdir(exist_ok=True, parents=True)
    with open(pareto_path, "wt") as f:
        for d in pareto_data:
            f.write(json.dumps(d) + "\n")
