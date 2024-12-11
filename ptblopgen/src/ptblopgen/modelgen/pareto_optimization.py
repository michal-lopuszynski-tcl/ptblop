import json
import logging

import numpy as np
import pymoo.algorithms.moo.nsga2
import pymoo.core.problem
import pymoo.operators.crossover.pntx
import pymoo.operators.mutation.bitflip
import pymoo.operators.sampling.rnd
import pymoo.optimize
import pymoo.termination

logger = logging.getLogger(__name__)


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
    n_attention = sum(int(layer["use_attention"]) for layer in bp_config.values())
    n_mlp = sum(int(layer["use_mlp"]) for layer in bp_config.values())
    return n, n_attention, n_mlp


# def main_tune_rf(args: argparse.Namespace):
#     pareto_dir = args.output_path / "pareto"
#     if pareto_dir.exists():
#         print(f"Output already exists, please delete it `rm -rf {pareto_dir}`")
#         sys.exit(1)
#     pareto_dir.mkdir(exist_ok=True, parents=True)

#     bp_config_db_path = args.output_path / DB_FNAME
#     data_trn, data_val = estimator_helpers.read_data(bp_config_db_path)
#     reg_kwargs_common = dict(n_regressors=20, n_estimators=300)

#     min_samples_leaf_values = [1, 2, 4, 10]
#     max_features_values = [0.1, 0.3, 0.5, 0.8, 1.0]

#     reg_kwargs_values = []

#     for min_samples_leaf in min_samples_leaf_values:
#         for max_features in max_features_values:
#             reg_kwargs = reg_kwargs_common | {
#                 "max_features": max_features,
#                 "min_samples_leaf": min_samples_leaf,
#             }
#             reg_kwargs_values.append(reg_kwargs)
#     import random

#     r = random.Random(42)
#     r.shuffle(reg_kwargs_values)

#     for i, reg_kwargs in enumerate(reg_kwargs_values, start=1):
#         regressor_dir = pareto_dir / f"{i:03d}"
#         regressor_dir.mkdir(parents=True, exist_ok=True)
#         reg_quality, reg_quality_data = quality_regressor_fit(
#             data_trn,
#             data_val,
#             target_column=TARGET_COLUMN,
#             regressor_dir=regressor_dir,
#             reg_kwargs=reg_kwargs,
#         )


def find_pareto_front(
    *,
    quality_estimator,
    quality_metric_name,
    cost_estimator,
    bp_config_unpruned,
    n_features,
    pareto_path,
):
    f_quality = build_predict_quality(quality_estimator)
    f_cost = build_predict_cost(cost_estimator)

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
        bp_config = get_bp_config_from_features(bp_config_unpruned, features)
        n, n_attention, n_mlp = get_bp_config_stats(bp_config)
        q, qmin, qmax = quality_estimator.predict_with_bounds([features])
        params = cost_estimator.predict([features])
        d = {
            "mparams_pred": params.item(),
            "n": n,
            "n_attention": n_attention,
            "n_mlp": n_mlp,
            f"{quality_metric_name}_pred": q.item(),
            f"{quality_metric_name}_pred_min": qmin.item(),
            f"{quality_metric_name}_pred_max": qmax.item(),
            "bp_config": bp_config,
        }
        pareto_data.append(d)
    pareto_data = sorted(pareto_data, key=lambda d: -d["mparams_pred"])
    pareto_path.parent.mkdir(exist_ok=True, parents=True)
    with open(pareto_path, "wt") as f:
        for d in pareto_data:
            f.write(json.dumps(d) + "\n")
