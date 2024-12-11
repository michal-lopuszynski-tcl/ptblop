import collections.abc
import copy
import datetime
import json
import logging
import pathlib
import random
from typing import Any

import numpy as np
import ptblop
import torch

from .. import _version, builders, estimators
from . import configurator, estimator_helpers, pareto_optimization

BPCONFIG_DB_FNAME = "bp_configs.json"
QUALITY_ESTIMATOR_REPORT_DIR = "quality_estimators"
QUALITY_ESTIMATOR_DB_FNAME = "quality_estimators.json"
QUALITY_ESTIMATOR_ID_TEMPLATE = "quality_estimator_%04d"
PARETO_FRONT_DIR = "pareto_fronts"
PARETO_FRONT_FNAME_TEMPLATE = "pareto_front_%04d.json"
STOP_FNAME = "STOP"

MAX_RANDOM_CONFIG_TRIALS = 20

logger = logging.getLogger(__name__)


# Helpers


def get_timestamp() -> str:
    current_utc = datetime.datetime.now(datetime.timezone.utc)
    return current_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_num_params(m: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def update_db(db_path, db_entry, mode="append"):
    if mode == "append":
        flag = "at"
    elif mode == "reset":
        flag = "wt"
    else:
        raise ValueError(f"Unknown mode {mode}")

    with open(db_path, flag) as f:
        f.write(json.dumps(db_entry) + "\n")


# Bpconfigs - helper functions


def genereate_bp_config_changes(bp_config_unpruned):
    config_changes = []

    for k, v in bp_config_unpruned.items():
        for k1 in v:
            config_changes.append({k: {k1: False}})
    return config_changes


def get_bp_config_signature(bp_config):
    singature_strs = []

    for v in bp_config.values():
        v_signature_str = str(int(not v["use_attention"])) + str(int(not v["use_mlp"]))
        singature_strs.append(v_signature_str)
    signature_str = "".join(singature_strs)
    return int(signature_str, 2)


def apply_bp_config_changes(bp_config_unpruned, bp_config_changes):
    blockprune_cfg = copy.deepcopy(bp_config_unpruned)

    for c in bp_config_changes:
        for k, v in c.items():
            blockprune_cfg[k] |= v

    return blockprune_cfg


def are_all_bp_configs_processed(bp_configs, bp_config_id_fmt, bp_config_db_path):
    if bp_config_db_path.exists():
        bp_config_ids = {
            (bp_config_id_fmt % i) for i, _ in enumerate(bp_configs, start=1)
        }
        processed_bp_configs_ids = set()

        with open(bp_config_db_path, "rt") as f:
            for line in f:
                d = json.loads(line)
                processed_bp_configs_ids.add(d["id"])

        n1 = len(bp_config_ids)
        n2 = len(bp_config_ids & processed_bp_configs_ids)
        logger.info(f"{bp_config_id_fmt} {n1=} {n2=}")
        return n1 == n2
    else:
        return False


def process_single_bp_config(*, model, device, bp_config_data, evaluator_fn):
    bp_config, bp_config_id, bp_config_score = bp_config_data
    res = {"id": bp_config_id}
    ptblop.apply_bp_config_in_place(model, bp_config, set_unused_layers_to_none=False)
    res["n_attention"] = ptblop.get_num_attention_blocks(model)
    res["n_mlp"] = ptblop.get_num_mlp_blocks(model)
    res["n"] = ptblop.get_num_prunable_blocks(model)
    res["mparams"] = ptblop.get_num_params(model) / 1.0e6
    res["evaluation"] = evaluator_fn(model, device)
    res["bp_config_score"] = bp_config_score
    res["bp_config"] = bp_config
    res["timestamp"] = get_timestamp()
    device_str = str(device)
    if "cuda" in device_str:
        device_str += " @ " + torch.cuda.get_device_name(device)
    res["device"] = device_str
    res["ptblop_version"] = ptblop.__version__
    res["ptblopgen_version"] = _version.__version__
    return res


def process_bp_configs(
    *,
    bp_configs,
    bp_config_scores,
    bp_config_id_prefix,
    bp_config_db_path,
    model,
    device,
    evaluator_fn,
    processed_bp_config_signatures,
    stop_path,
):
    assert len(bp_configs) == len(bp_config_scores)
    bp_config_id_fmt = f"{bp_config_id_prefix}%04d"

    if are_all_bp_configs_processed(bp_configs, bp_config_id_fmt, bp_config_db_path):
        logger.info(f"All configs from batch {bp_config_id_prefix} already processed")
    else:
        bp_configs_and_scores = zip(bp_configs, bp_config_scores)
        for i, (bp_config, bp_config_score) in enumerate(
            bp_configs_and_scores, start=1
        ):
            if stop_path.exists():
                logger.warning(f"Stop file found {stop_path}, exiting...")
                break
            bp_config_signature = get_bp_config_signature(bp_config)
            if bp_config_signature in processed_bp_config_signatures:
                logger.warning(f"Model already processed {bp_config_signature=}")
            else:
                bp_config_id = bp_config_id_fmt % i
                bp_config_data = (bp_config, bp_config_id, bp_config_score)
                res = process_single_bp_config(
                    bp_config_data=bp_config_data,
                    model=model,
                    device=device,
                    evaluator_fn=evaluator_fn,
                )
                assert bp_config_signature not in processed_bp_config_signatures
                processed_bp_config_signatures.add(bp_config_signature)
                update_db(bp_config_db_path, res)


def read_processed_bp_config_signatures(db_path):
    signatures = set()

    with open(db_path, "rt") as f:
        for line in f:
            d = json.loads(line)
            bp_config = d["bp_config"]
            sig = get_bp_config_signature(bp_config)
            signatures.add(sig)
    logger.info(f"Read {len(signatures)} configurations and stored their signatures")
    return signatures


# Bpconfigs - generation - one layer


def make_one_layer_bp_configs(bp_config_unpruned):
    cfg_changes_all = genereate_bp_config_changes(bp_config_unpruned)
    res = []

    for cfg_change in cfg_changes_all:
        bp_config = copy.deepcopy(bp_config_unpruned)
        bp_config = apply_bp_config_changes(bp_config, [cfg_change])
        res.append(bp_config)

    return res, [-1.0 for r in res]


# Bpconfigs - generation - random


def _make_random_bp_config(
    *,
    rng,
    bp_config_unpruned,
    num_changes,
    processed_bp_config_signatures,
    cfg_changes_all=None,
    max_random_config_trials,
):
    # TODO Delete this line
    if cfg_changes_all is None:
        cfg_changes_all = genereate_bp_config_changes(bp_config_unpruned)

    bp_config, bp_config_signature = None, None

    for j in range(1, max_random_config_trials + 1):
        cfg_changes = rng.sample(cfg_changes_all, k=num_changes)
        bp_config = apply_bp_config_changes(bp_config_unpruned, cfg_changes)
        bp_config_signature = get_bp_config_signature(bp_config)

        if bp_config_signature not in processed_bp_config_signatures:
            break
        else:
            msg = f"Try {j=}, drew signature={bp_config_signature}"
            msg += " that is already processed, repeating"
            logger.warning(msg)

    return bp_config, bp_config_signature


def make_random_bp_configs(
    *,
    bp_config_unpruned,
    n,
    rng,
    min_num_changes,
    max_num_changes,
    max_random_config_trials,
    processed_bp_bconfig_signatures,
):
    processed_bpbconfig_signatures_all = copy.deepcopy(processed_bp_bconfig_signatures)
    cfg_changes_all = genereate_bp_config_changes(bp_config_unpruned)

    res = []
    while len(res) < n:

        num_changes = rng.randint(min_num_changes, max_num_changes)
        bp_config, bp_config_signature = _make_random_bp_config(
            rng=rng,
            bp_config_unpruned=bp_config_unpruned,
            num_changes=num_changes,
            processed_bp_config_signatures=processed_bpbconfig_signatures_all,
            cfg_changes_all=cfg_changes_all,
            max_random_config_trials=max_random_config_trials,
        )
        if bp_config is not None:
            res.append(bp_config)
            processed_bpbconfig_signatures_all.add(bp_config_signature)
    res_scores = [-1.0 for r in res]  # For random configs scores are meaningless
    return res, res_scores


# Bp configs - generation - with scoring


def make_random_bp_configs_with_scoring(
    *,
    bp_config_unpruned,
    num_configs,
    rng,
    min_num_changes,
    max_num_changes,
    processed_bpbconfig_signatures,
    num_scored_candidates,
    scoring_fn,
):
    processed_bpbconfig_signatures_all = copy.deepcopy(processed_bpbconfig_signatures)
    cfg_changes_all = genereate_bp_config_changes(bp_config_unpruned)

    final_bp_configs = []
    final_scores = []
    while len(final_bp_configs) < num_configs:

        num_changes = rng.randint(min_num_changes, max_num_changes)

        # Generate candidates

        candidate_bp_configs = []
        candidate_signatures = set()
        for _ in range(num_scored_candidates):
            tmp_singnatures = processed_bpbconfig_signatures_all | candidate_signatures
            bp_config, bp_config_signature = _make_random_bp_config(
                rng=rng,
                bp_config_unpruned=bp_config_unpruned,
                num_changes=num_changes,
                processed_bp_config_signatures=tmp_singnatures,
                cfg_changes_all=cfg_changes_all,
            )
            if bp_config is not None:
                candidate_bp_configs.append(bp_config)
                candidate_signatures.add(bp_config_signature)

        # Select highest scored candidate

        if len(candidate_bp_configs) > 0:
            n_c = len(candidate_bp_configs)
            logger.info(f"Generated {n_c} config candidates, selecting one config")
            scores = np.array([scoring_fn(bpc) for bpc in candidate_bp_configs])
            imax = np.argmax(scores)
            bp_config = candidate_bp_configs[imax]
            max_score = scores[imax]
            min_score = np.min(scores)
            mean_score = np.mean(scores)
            assert max_score == scoring_fn(bp_config)

            logger.info(f"Score stats: {max_score=} {min_score=} {mean_score=}")
            # scores = [scoring_fn(bpc) for bpc in candidate_bp_configs]
            # scores.sort(reverse=True)

            # for i, r in enumerate(scores, start=1):
            #     logger.info(f"Scoring {i} - {r}")

            bp_config_signature = get_bp_config_signature(bp_config)
            final_bp_configs.append(bp_config)
            final_scores.append(max_score)
            processed_bpbconfig_signatures_all.add(bp_config_signature)
        else:
            logger.info(f"Failed to generate any configs for {num_changes=}")

    return final_bp_configs, final_scores


# Bpconfigs - predictor based scoring


def conv_regressor_to_scoring_fn(reg):

    def __scoring_fn(bp_config):
        X = estimator_helpers.get_quality_features([bp_config])
        _, ypred_min, ypred_max = reg.predict_with_bounds(X)
        return np.abs(ypred_max - ypred_min).item()

    return __scoring_fn


def make_scoring_fn(regressor_id, bp_config_db_path, regressor_db_path):
    data_trn, data_val = estimator_helpers.read_data(bp_config_db_path)

    X_trn = estimator_helpers.get_quality_features([d["bp_config"] for d in data_trn])
    y_trn = estimator_helpers.get_target(data_trn, "arc_challenge_acc")
    logger.info(f"{X_trn.shape=} {X_trn.dtype=} {y_trn.shape=} {y_trn.dtype=}")

    X_val = estimator_helpers.get_quality_features([d["bp_config"] for d in data_val])
    y_val = estimator_helpers.get_target(data_val, "arc_challenge_acc")
    logger.info(f"{X_val.shape=} {X_val.dtype=} {y_val.shape=} {y_val.dtype=}")

    n_examples_trn, n_features_trn = X_trn.shape
    n_examples_val, n_features_val = X_val.shape

    reg_type = "QuantileGradientBoostingBoundsRegressor"
    reg_kwargs = dict(
        learning_rate=0.03,
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=9,
        min_samples_split=9,
    )

    reg = estimators.QuantileGradientBoostingBoundsEstimator(**reg_kwargs)
    reg.fit(X_trn, y_trn)
    reg_metrics = estimator_helpers.evaluate_bounds_estimator(
        bounds_regressor=reg, X_trn=X_trn, y_trn=y_trn, X_val=X_val, y_val=y_val
    )
    logger.info(f"{reg_metrics=}")

    scoring_fn = conv_regressor_to_scoring_fn(reg)
    db_entry = {
        "regressor_id": regressor_id,
        "n_examples_trn": n_examples_trn,
        "n_features_trn": n_features_trn,
        "n_examples_val": n_examples_val,
        "n_features_val": n_features_val,
        "regressor_type": reg_type,
        "regressor_kwargs": reg_kwargs,
        "regressor_metrics": reg_metrics,
        "blockprunekit_version": _version.__version__,
    }
    update_db(regressor_db_path, db_entry)
    return scoring_fn


def bp_configs_scores_sample(bp_configs, bp_config_scores, n: int, rng):
    if n < 0 or len(bp_configs) < n:
        return bp_configs, bp_config_scores
    else:
        indices = rng.sample(range(len(bp_configs)), k=n)
        bp_configs_new = [bp_configs[i] for i in indices]
        bp_config_scores_new = [bp_config_scores[i] for i in indices]
        return bp_configs_new, bp_config_scores_new


def gen_sample_configs(
    trn_schedule: list[configurator.TrnSchedulerEntryConfig],
) -> collections.abc.Generator[tuple[int, int, int, int], None, None]:

    for t in trn_schedule:
        yield t.n_trn_onel, t.n_trn_rand, t.n_trn_actl, t.n_trn_parf

    t = trn_schedule[-1]

    while True:
        yield t.n_trn_onel, t.n_trn_rand, t.n_trn_actl, t.n_trn_parf


def sample_one_layer_bp_configs(
    *,
    model,
    device,
    evaluator_fn,
    n,
    bp_config_id_prefix,
    processed_bp_config_signatures,
    rng,
    bp_config_db_path,
    stop_path,
):
    bp_config_unpruned = ptblop.get_unpruned_bp_config(model)

    bp_configs, bp_config_scores = make_one_layer_bp_configs(bp_config_unpruned)

    bp_configs, bp_config_scores = bp_configs_scores_sample(
        bp_configs, bp_config_scores, n, rng
    )
    process_bp_configs(
        bp_configs=bp_configs,
        bp_config_scores=bp_config_scores,
        bp_config_id_prefix=bp_config_id_prefix,
        bp_config_db_path=bp_config_db_path,
        model=model,
        evaluator_fn=evaluator_fn,
        device=device,
        processed_bp_config_signatures=processed_bp_config_signatures,
        stop_path=stop_path,
    )


def sample_random_bp_configs(
    *,
    model,
    device,
    evaluator_fn,
    n,
    max_num_changes,
    bp_config_id_prefix,
    processed_bp_config_signatures,
    rng,
    bp_config_db_path,
    stop_path,
) -> None:
    bp_config_unpruned = ptblop.get_unpruned_bp_config(model)
    bp_configs, bp_config_scores = make_random_bp_configs(
        bp_config_unpruned=bp_config_unpruned,
        n=n,
        rng=rng,
        min_num_changes=2,
        max_num_changes=max_num_changes,
        max_random_config_trials=MAX_RANDOM_CONFIG_TRIALS,
        processed_bp_bconfig_signatures=processed_bp_config_signatures,
    )

    process_bp_configs(
        bp_configs=bp_configs,
        bp_config_scores=bp_config_scores,
        bp_config_id_prefix=bp_config_id_prefix,
        bp_config_db_path=bp_config_db_path,
        model=model,
        evaluator_fn=evaluator_fn,
        device=device,
        processed_bp_config_signatures=processed_bp_config_signatures,
        stop_path=stop_path,
    )


# def make_scoring_fn(regressor_id, bp_config_db_path, regressor_db_path):
#     data_trn, data_val = estimator_helpers.read_data(bp_config_db_path)

#     X_trn = estimator_helpers.get_quality_features([d["bp_config"] for d in data_trn])
#     y_trn = estimator_helpers.get_target(data_trn, "arc_challenge_acc")
#     logger.info(f"{X_trn.shape=} {X_trn.dtype=} {y_trn.shape=} {y_trn.dtype=}")

#     X_val = estimator_helpers.get_quality_features([d["bp_config"] for d in data_val])
#     y_val = estimator_helpers.get_target(data_val, "arc_challenge_acc")
#     logger.info(f"{X_val.shape=} {X_val.dtype=} {y_val.shape=} {y_val.dtype=}")

#     n_examples_trn, n_features_trn = X_trn.shape
#     n_examples_val, n_features_val = X_val.shape

#     reg_type = "QuantileGradientBoostingBoundsRegressor"
#     reg_kwargs = dict(
#         learning_rate=0.03,
#         n_estimators=200,
#         max_depth=4,
#         min_samples_leaf=9,
#         min_samples_split=9,
#     )

#     reg = estimators.QuantileGradientBoostingBoundsEstimator(**reg_kwargs)
#     reg.fit(X_trn, y_trn)
#     reg_metrics = estimator_helpers.evaluate_bounds_regressor(
#         bounds_regressor=reg, X_trn=X_trn, y_trn=y_trn, X_val=X_val, y_val=y_val
#     )
#     logger.info(f"{reg_metrics=}")

#     scoring_fn = conv_regressor_to_scoring_fn(reg)
#     db_entry = {
#         "regressor_id": regressor_id,
#         "n_examples_trn": n_examples_trn,
#         "n_features_trn": n_features_trn,
#         "n_examples_val": n_examples_val,
#         "n_features_val": n_features_val,
#         "regressor_type": reg_type,
#         "regressor_kwargs": reg_kwargs,
#         "regressor_metrics": reg_metrics,
#         "blockprunekit_version": _version.__version__,
#     }
#     update_db(regressor_db_path, db_entry)
#     return scoring_fn


# def sample_active_learning_bp_configs(
#     *,
#     model,
#     device,
#     evaluator_fn,
#     n,
#     max_num_changes,
#     bp_config_id_prefix,
#     processed_bp_config_signatures,
#     rng,
#     bp_config_db_path,
#     stop_path,
#     n_candidates,
#     estimator_db_path
# ):
#     bp_config_unpruned = ptblop.get_unpruned_bp_config(model)
#     scoring_fn = make_scoring_fn(
#         bp_config_prefix, bp_config_db_path, estimator_db_path
#     )
#     bp_configs, bp_config_scores = make_random_bp_configs_with_scoring(
#         bp_config_unpruned=bp_config_unpruned,
#         num_configs=n,
#         rng=rng,
#         min_num_changes=2,
#         max_num_changes=max_num_changes,
#         processed_bpbconfig_signatures=processed_bp_config_signatures,
#         num_scored_candidates=n_candidates,
#         scoring_fn=scoring_fn,
#     )
#     process_bp_configs(
#         bp_configs=bp_configs,
#         bp_config_scores=bp_config_scores,
#         bp_config_id_prefix=bp_config_id_prefix,
#         bp_config_db_path=bp_config_db_path,
#         model=model,
#         evaluator_fn=evaluator_fn,
#         device=device,
#         processed_bp_config_signatures=processed_bp_config_signatures,
#         stop_path=stop_path,
#     )


def sample_active_learning_bp_configs(
    *,
    model,
    device,
    evaluator_fn,
    n,
    max_num_changes,
    bp_config_id_prefix,
    processed_bp_config_signatures,
    rng,
    bp_config_db_path,
    stop_path,
    n_candidates,
    estimator_db_path,
):
    pass


def sample_pareto_front_bp_configs(
    *,
    model,
    device,
    evaluator_fn,
    n,
    max_num_changes,
    bp_config_id_prefix,
    processed_bp_config_signatures,
    rng,
    bp_config_db_path,
    stop_path,
):
    pass


def main_modelgen(config: dict[str, Any], output_path: pathlib.Path) -> None:
    bp_config_db_path = output_path / BPCONFIG_DB_FNAME
    quality_estimators_db_path = output_path / QUALITY_ESTIMATOR_DB_FNAME
    quality_estimators_report_path = output_path / QUALITY_ESTIMATOR_REPORT_DIR
    stop_path = output_path / STOP_FNAME

    quality_estimators_report_path.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, evaluator_fn = builders.make_model_and_evaluator(
        config["model"], config["evaluator"], device
    )

    bp_config_unpruned = ptblop.get_unpruned_bp_config(model)
    ptblop.apply_bp_config_in_place(model, {})

    config_sampler = configurator.SamplerConfig(**config["sampler"])

    rng = random.Random(config_sampler.random_bp_config_rng_seed)

    max_num_changes = round(
        config_sampler.max_num_changes_factor * len(bp_config_unpruned)
    )
    logger.info(
        f"{max_num_changes=} num_blocks={len(bp_config_unpruned)} "
        f"max_num_changes_factor={config_sampler.max_num_changes_factor}"
    )

    processed_bp_config_signatures = set()

    fixed_kwargs = dict(
        bp_config_db_path=bp_config_db_path,
        stop_path=stop_path,
        device=device,
        model=model,
        evaluator_fn=evaluator_fn,
        rng=rng,
        processed_bp_config_signatures=processed_bp_config_signatures,
    )

    # VALIDATION DATASET

    sample_random_bp_configs(
        n=config_sampler.n_val_rand,
        bp_config_id_prefix="val.rndl.001.",
        max_num_changes=max_num_changes,
        **fixed_kwargs,
    )

    # TRAINING DATASET - UNPRUNED MODEL

    process_bp_configs(
        bp_configs=[bp_config_unpruned],
        bp_config_scores=[-1],
        bp_config_id_prefix="trn.zerl.001.",
        bp_config_db_path=bp_config_db_path,
        model=model,
        evaluator_fn=evaluator_fn,
        device=device,
        processed_bp_config_signatures=processed_bp_config_signatures,
        stop_path=stop_path,
    )

    # TRAINING DATASET - ITERATIONS AS DEFINED IN CONFIG

    for i, n_tuple in enumerate(
        gen_sample_configs(config_sampler.trn_schedule), start=1
    ):
        n_onel, n_rand, n_actl, n_parf = n_tuple

        sample_one_layer_bp_configs(
            n=n_onel,
            bp_config_id_prefix=f"trn.onel.{i:04d}.",
            **fixed_kwargs,
        )

        sample_random_bp_configs(
            n=n_rand,
            max_num_changes=max_num_changes,
            bp_config_id_prefix=f"trn.rand.{i:04d}.",
            **fixed_kwargs,
        )
        # sample_active_learning_bp_configs(
        #     n=n_actl,
        #     bp_config_id_prefix=f"trn.actl.{i:04d}.",
        #     **fixed_kwargs,
        # )

        # sample_pareto_front_bp_configs(
        #     n=n_parf,
        #     bp_config_id_prefix=f"trn.actl.{i:04d}.",
        #     **fixed_kwargs,
        # )

        if i == 1:
            cost_estimator, cost_estimator_metrics = (
                estimator_helpers.train_param_estimator(bp_config_db_path)
            )

        quality_estimator_id = "quality_estimator_%04d" % i
        quality_estimator, quality_estimator_metrics = (
            estimator_helpers.train_quality_estimator(
                bp_config_db_path=bp_config_db_path,
                quality_metric=config_sampler.quality_evaluator_metric,
                quality_estimator_id=quality_estimator_id,
                quality_estimator_report_path=quality_estimators_report_path,
            )
        )
        update_db(quality_estimators_db_path, quality_estimator_metrics)
        n_features = quality_estimator_metrics["n_features_trn"]
        pareto_front_path = (
            output_path / PARETO_FRONT_DIR / (PARETO_FRONT_FNAME_TEMPLATE % i)
        )
        pareto_optimization.find_pareto_front(
            quality_estimator=quality_estimator,
            quality_metric_name=config_sampler.quality_evaluator_metric,
            cost_estimator=cost_estimator,
            n_features=n_features,
            bp_config_unpruned=bp_config_unpruned,
            pareto_path=pareto_front_path,
        )
