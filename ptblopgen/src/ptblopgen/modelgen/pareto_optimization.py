import json
import logging
import time

import numpy as np
import pymoo.algorithms.moo.nsga2
import pymoo.core.problem
import pymoo.core.sampling
import pymoo.operators.crossover.pntx
import pymoo.operators.mutation.bitflip
import pymoo.optimize
import pymoo.termination

from .. import utils
from . import configurator

QUALITY_ESTIMATOR_REPORT_DIR = "estimators_quality"
QUALITY_ESTIMATOR_DB_FNAME = "estimators_quality.json"
COST_ESTIMATOR_DB_FNAME = "estimators_cost.json"
PARETO_FRONT_DIR = "pareto_fronts"
PARETO_FRONT_FNAME_TEMPLATE = "pareto_front_%04d.json"


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


class BinomialSampling(pymoo.core.sampling.Sampling):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < self.p).astype(bool)


class UniformSampling(pymoo.core.sampling.Sampling):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def _do(self, problem, n_samples, **kwargs):
        min_n_var = int(problem.n_var * self.p)
        row_sequence = np.arange(problem.n_var, dtype=float)
        mat_range = np.tile(row_sequence, (n_samples, 1))
        r = np.random.randint(low=min_n_var, high=problem.n_var, size=(n_samples, 1))
        mask = mat_range < r
        idx = np.random.rand(*mask.shape).argsort(axis=1)
        rows = np.arange(mask.shape[0])[:, None]
        return mask[rows, idx]


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


def get_bp_config_from_features(sample_bp_config, features, full_block_mode):

    res = {}
    if not full_block_mode:
        assert len(features) == 2 * len(sample_bp_config)
        for i, k in enumerate(sample_bp_config):
            use_attention = float(features[2 * i]) > 0.5
            use_mlp = float(features[2 * i + 1]) > 0.5
            res[k] = {"use_attention": use_attention, "use_mlp": use_mlp}
    else:
        assert len(features) == len(sample_bp_config)
        for i, k in enumerate(sample_bp_config):
            is_block_on = float(features[i]) > 0.5
            res[k] = {"use_attention": is_block_on, "use_mlp": is_block_on}

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
    run_id,
    model_metadata,
    quality_estimator,
    quality_estimator_id,
    quality_metric_name,
    cost_estimator,
    cost_estimator_id,
    bp_config_unpruned,
    n_features,
    pareto_path,
    config_pareto_optimization: configurator.ParetoOptimizationConfig,
    full_block_mode,
):
    t1 = time.perf_counter()
    f_quality = build_predict_quality(quality_estimator)
    f_cost = build_predict_cost(cost_estimator)

    problem = BinaryProblem(f1=f_quality, f2=f_cost, n_var=n_features)

    sampling_mode = config_pareto_optimization.sampling_mode
    logger.info(f"Using {sampling_mode=} in Pareto optimization")
    if sampling_mode == "binomial":
        sampling = BinomialSampling(config_pareto_optimization.sampling_p)
    elif sampling_mode == "uniform":
        sampling = UniformSampling(config_pareto_optimization.sampling_p)
    else:
        msg = f"Unknown {sampling_mode=} only 'uniform' or 'binomial' are supported"
        raise ValueError(msg)

    algorithm = pymoo.algorithms.moo.nsga2.NSGA2(
        pop_size=config_pareto_optimization.pop_size,
        sampling=sampling,
        crossover=pymoo.operators.crossover.pntx.TwoPointCrossover(),
        mutation=pymoo.operators.mutation.bitflip.BitflipMutation(prob=1 / n_features),
        eliminate_duplicates=True,
    )

    termination = pymoo.termination.get_termination(
        "n_gen", config_pareto_optimization.n_gen
    )

    res = pymoo.optimize.minimize(
        problem,
        algorithm,
        termination,
        seed=config_pareto_optimization.optimizer_seed,
        verbose=True,
    )

    pareto_front = res.F

    # np.savetxt(pareto_dir / "pareto_front.csv", pareto_front, delimiter=",")
    # np.savetxt(pareto_dir / "pareto_solutions.csv", res.X, delimiter=",", fmt="%d")
    m, _ = pareto_front.shape

    pareto_data = []
    ts = utils.get_timestamp()
    v_ptblop, v_ptblopgen = utils.get_versions()
    for i in range(m):
        features = res.X[i, :]
        bp_config = get_bp_config_from_features(
            bp_config_unpruned, features, full_block_mode
        )
        n, n_attention, n_mlp = get_bp_config_stats(bp_config)
        q, qmin, qmax = quality_estimator.predict_with_bounds([features])
        params = cost_estimator.predict([features])
        d = {
            "run_id": run_id,
            "mparams_pred": params.item(),
            "n": n,
            "n_attention": n_attention,
            "n_mlp": n_mlp,
            f"{quality_metric_name}_pred": q.item(),
            f"{quality_metric_name}_pred_min": qmin.item(),
            f"{quality_metric_name}_pred_max": qmax.item(),
            "cost_estimator_id": cost_estimator_id,
            "quality_estimator_id": quality_estimator_id,
            "timestamp": ts,
            "ptblop_version": v_ptblop,
            "ptblopgen_version": v_ptblopgen,
            "bp_config_signature": hex(utils.get_bp_config_signature(bp_config))[2:],
            "model_metadata": model_metadata,
            "bp_config": bp_config,
        }
        pareto_data.append(d)
    pareto_data = sorted(pareto_data, key=lambda d: -d["mparams_pred"])
    pareto_path.parent.mkdir(exist_ok=True, parents=True)
    with open(pareto_path, "wt") as f:
        for d in pareto_data:
            f.write(json.dumps(d) + "\n")
    t2 = time.perf_counter()
    msg = f"Finished Pareto optimization: duration={t2-t1:.2f} seconds"
    msg += f" n_gen={config_pareto_optimization.n_gen}"
    msg += f" pos_size={config_pareto_optimization.pop_size}"
    msg += f" optimizer_seed={config_pareto_optimization.optimizer_seed}"
    msg += f" n_features={n_features}"
    logger.info(msg)


def get_one_indices(cfg):
    return [i for i, c in enumerate(cfg) if c == 1]


def get_cfg_zeroed_at(cfg, zero_index):
    cfg_list = list(cfg)
    cfg_list[zero_index] = 0
    return bytes(cfg_list)


def gen_new_cfgs_single(cfg, processed_cfgs):
    new_configurations = []
    for i in get_one_indices(cfg):
        cfg_new = get_cfg_zeroed_at(cfg, i)
        if cfg_new not in processed_cfgs:
            new_configurations.append(cfg_new)
            processed_cfgs.add(cfg_new)
    return new_configurations


def gen_new_cfgs(cfgs, processed_cfgs):
    new_configurations = []
    for cfg in cfgs:
        new_configurations.extend(gen_new_cfgs_single(cfg, processed_cfgs))
    return new_configurations


def rank_cfgs(cfgs, quality_estimator):
    logger.info(f"Started ranking n={len(cfgs)} configurations")
    X = [list(cfg) for cfg in cfgs]
    # logger.info(f"{X=}")
    X = np.array(X, dtype=np.float32)
    quality, _, _ = quality_estimator.predict_with_bounds(X)
    logger.info(f"Finihsed ranking n={len(cfgs)} configurations")
    return np.argsort(-quality)


def find_pareto_front_beam(
    *,
    run_id,
    model_metadata,
    quality_estimator,
    quality_estimator_id,
    quality_metric_name,
    cost_estimator,
    cost_estimator_id,
    bp_config_unpruned,
    n_features,
    pareto_path,
    config_pareto_optimization: configurator.ParetoOptimizationConfig,
    full_block_mode,
):
    t1 = time.perf_counter()
    # f_quality = build_predict_quality(quality_estimator)
    # f_cost = build_predict_cost(cost_estimator)

    BEAM_SIZE = 30
    PARETO_SIZE = 1

    beam_cfgs = [bytes([1] * n_features)]
    processed_cfgs = set()

    pareto_cfgs = []

    for i in range(1, 20 + 1):
        candidate_cfgs = gen_new_cfgs(beam_cfgs, processed_cfgs)
        logger.info(f"{i=} generated n={len(candidate_cfgs)} candidate configs")

        ranking = rank_cfgs(candidate_cfgs, quality_estimator)

        beam_cfgs = [candidate_cfgs[i] for i in ranking[:BEAM_SIZE]]
        pareto_cfgs.extend([candidate_cfgs[i] for i in ranking[:PARETO_SIZE]])

    pareto_data = []
    ts = utils.get_timestamp()
    v_ptblop, v_ptblopgen = utils.get_versions()
    for i in range(len(pareto_cfgs)):
        features = np.array(list(pareto_cfgs[i]), dtype=np.float32)
        bp_config = get_bp_config_from_features(
            bp_config_unpruned, features, full_block_mode
        )
        n, n_attention, n_mlp = get_bp_config_stats(bp_config)
        q, qmin, qmax = quality_estimator.predict_with_bounds([features])
        params = cost_estimator.predict([features])
        d = {
            "run_id": run_id,
            "mparams_pred": params.item(),
            "n": n,
            "n_attention": n_attention,
            "n_mlp": n_mlp,
            f"{quality_metric_name}_pred": q.item(),
            f"{quality_metric_name}_pred_min": qmin.item(),
            f"{quality_metric_name}_pred_max": qmax.item(),
            "cost_estimator_id": cost_estimator_id,
            "quality_estimator_id": quality_estimator_id,
            "timestamp": ts,
            "ptblop_version": v_ptblop,
            "ptblopgen_version": v_ptblopgen,
            "bp_config_signature": hex(utils.get_bp_config_signature(bp_config))[2:],
            "model_metadata": model_metadata,
            "bp_config": bp_config,
        }
        pareto_data.append(d)
    pareto_data = sorted(pareto_data, key=lambda d: -d["mparams_pred"])
    pareto_path.parent.mkdir(exist_ok=True, parents=True)
    with open(pareto_path, "wt") as f:
        for d in pareto_data:
            f.write(json.dumps(d) + "\n")
    t2 = time.perf_counter()
    msg = f"Finished Pareto optimization: duration={t2-t1:.2f} seconds"
    msg += f" n_gen={config_pareto_optimization.n_gen}"
    msg += f" pos_size={config_pareto_optimization.pop_size}"
    msg += f" optimizer_seed={config_pareto_optimization.optimizer_seed}"
    msg += f" n_features={n_features}"
    logger.info(msg)
