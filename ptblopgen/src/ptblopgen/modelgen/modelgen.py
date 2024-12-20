import collections.abc
import copy
import json
import logging
import pathlib
import random
from typing import Any

import numpy as np
import ptblop
import torch

from .. import builders, estimators, utils
from . import configurator, estimator_helpers, pareto_optimization

BP_CONFIG_DB_FNAME = "bp_configs.json"
QUALITY_ESTIMATOR_REPORT_DIR = "estimators_quality"
QUALITY_ESTIMATOR_DB_FNAME = "estimators_quality.json"
COST_ESTIMATOR_DB_FNAME = "estimators_cost.json"
QUALITY_ESTIMATOR_ID_TEMPLATE = "estimator_quality_%04d"

PARETO_FRONT_DIR = "pareto_fronts"
PARETO_FRONT_FNAME_TEMPLATE = "pareto_front_%04d.json"
STOP_FNAME = "STOP"

MAX_RANDOM_CONFIG_TRIALS = 20

logger = logging.getLogger(__name__)


# Helpers


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
    bp_config = copy.deepcopy(bp_config_unpruned)

    for c in bp_config_changes:
        for k, v in c.items():
            bp_config[k] |= v

    return bp_config


# def are_all_bp_configs_processed(bp_configs, bp_config_id_fmt, bp_config_db_path):
#     if bp_config_db_path.exists():
#         bp_config_ids = {
#             (bp_config_id_fmt % i) for i, _ in enumerate(bp_configs, start=1)
#         }
#         processed_bp_configs_ids = set()

#         with open(bp_config_db_path, "rt") as f:
#             for line in f:
#                 d = json.loads(line)
#                 processed_bp_configs_ids.add(d["id"])

#         n1 = len(bp_config_ids)
#         n2 = len(bp_config_ids & processed_bp_configs_ids)
#         # logger.info(f"{bp_config_id_fmt} {n1=} {n2=}")
#         return n1 == n2
#     else:
#         return False


def process_bp_config(
    *,
    bp_config_id,
    bp_config_type,
    bp_config,
    bp_config_score,
    data_iter,
    bp_config_db_path,
    model,
    device,
    evaluator_fn,
    processed_bp_config_signatures,
    stop_path,
):
    if stop_path.exists():
        logger.warning(f"Stop file found {stop_path}, exiting...")
        return
    bp_config_signature = get_bp_config_signature(bp_config)
    if bp_config_signature in processed_bp_config_signatures:
        logger.warning(f"Model already processed {bp_config_signature=}")
    else:
        res = {"id": bp_config_id, "type": bp_config_type, "data_iter": data_iter}
        ptblop.apply_bp_config_in_place(
            model, bp_config, set_unused_layers_to_none=False
        )
        res["n_attention"] = ptblop.get_num_attention_blocks(model)
        res["n_mlp"] = ptblop.get_num_mlp_blocks(model)
        res["n"] = ptblop.get_num_prunable_blocks(model)
        res["mparams"] = ptblop.get_num_active_params(model) / 1.0e6
        res["evaluation"] = evaluator_fn(model, device)
        res["bp_config_score"] = bp_config_score
        res["bp_config"] = bp_config
        res["timestamp"] = utils.get_timestamp()
        device_str = str(device)
        if "cuda" in device_str:
            device_str += " @ " + torch.cuda.get_device_name(device)
        res["device"] = device_str
        v_ptblop, v_ptblopgen = utils.get_versions()
        res["ptblop_version"] = v_ptblop
        res["ptblopgen_version"] = v_ptblopgen

        assert bp_config_signature not in processed_bp_config_signatures
        processed_bp_config_signatures.add(bp_config_signature)
        update_db(bp_config_db_path, res)


# def read_processed_bp_config_signatures(db_path):
#     signatures = set()

#     with open(db_path, "rt") as f:
#         for line in f:
#             d = json.loads(line)
#             bp_config = d["bp_config"]
#             sig = get_bp_config_signature(bp_config)
#             signatures.add(sig)
#     logger.info(f"Read {len(signatures)} configurations and stored their signatures")
#     return signatures


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
    # TODO Merge this with make_random_bp_config

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
            msg = f"Try {j=}, {num_changes=},  drew signature={bp_config_signature}"
            msg += " that is already processed, repeating"
            logger.warning(msg)

    return bp_config, bp_config_signature


def make_random_bp_config(
    *,
    bp_config_unpruned,
    rng,
    min_num_changes,
    max_num_changes,
    max_random_config_trials,
    processed_bp_bconfig_signatures,
):
    # TODO Merge this with _make_random_bp_config

    processed_bpbconfig_signatures_all = copy.deepcopy(processed_bp_bconfig_signatures)
    cfg_changes_all = genereate_bp_config_changes(bp_config_unpruned)

    bp_config = None

    while bp_config is None:
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
            processed_bpbconfig_signatures_all.add(bp_config_signature)
    bp_config_score = -1  # For random configs scores are meaningless
    return bp_config, bp_config_score


# Bp configs - generation - with scoring


# def make_random_bp_configs_with_scoring(
#     *,
#     bp_config_unpruned,
#     n,
#     rng,
#     min_num_changes,
#     max_num_changes,
#     processed_bp_bconfig_signatures,
#     max_random_config_trials,
#     num_scored_candidates,
#     scoring_fn,
# ):
#     processed_bpbconfig_signatures_all = copy.deepcopy(processed_bp_bconfig_signatures)
#     cfg_changes_all = genereate_bp_config_changes(bp_config_unpruned)

#     final_bp_configs = []
#     final_scores = []
#     while len(final_bp_configs) < n:

#         num_changes = rng.randint(min_num_changes, max_num_changes)

#         # Generate candidates

#         candidate_bp_configs = []
#         candidate_signatures = set()
#         for _ in range(num_scored_candidates):
#             tmp_singnatures = processed_bpbconfig_signatures_all | candidate_signatures
#             bp_config, bp_config_signature = _make_random_bp_config(
#                 rng=rng,
#                 bp_config_unpruned=bp_config_unpruned,
#                 num_changes=num_changes,
#                 processed_bp_config_signatures=tmp_singnatures,
#                 cfg_changes_all=cfg_changes_all,
#                 max_random_config_trials=max_random_config_trials,
#             )
#             if bp_config is not None:
#                 candidate_bp_configs.append(bp_config)
#                 candidate_signatures.add(bp_config_signature)

#         # Select highest scored candidate

#         if len(candidate_bp_configs) > 0:
#             n_c = len(candidate_bp_configs)
#             logger.info(f"Generated {n_c} config candidates, selecting one config")
#             scores = np.array([scoring_fn(bpc) for bpc in candidate_bp_configs])
#             imax = np.argmax(scores)
#             bp_config = candidate_bp_configs[imax]
#             max_score = scores[imax]
#             min_score = np.min(scores)
#             mean_score = np.mean(scores)
#             assert max_score == scoring_fn(bp_config)

#             logger.info(f"Score stats: {max_score=} {min_score=} {mean_score=}")
#             # scores = [scoring_fn(bpc) for bpc in candidate_bp_configs]
#             # scores.sort(reverse=True)

#             # for i, r in enumerate(scores, start=1):
#             #     logger.info(f"Scoring {i} - {r}")

#             bp_config_signature = get_bp_config_signature(bp_config)
#             final_bp_configs.append(bp_config)
#             final_scores.append(max_score)
#             processed_bpbconfig_signatures_all.add(bp_config_signature)
#         else:
#             logger.info(f"Failed to generate any configs for {num_changes=}")

#     return final_bp_configs, final_scores


# Bpconfigs - predictor based scoring


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
#     reg_metrics = estimator_helpers.evaluate_bounds_estimator(
#         bounds_regressor=reg, X_trn=X_trn, y_trn=y_trn, X_val=X_val, y_val=y_val
#     )
#     logger.info(f"{reg_metrics=}")

#     scoring_fn = conv_quality_estimator_to_scoring_fn(reg)

#     v_ptblop, v_ptblopben = utils.get_versions()

#     db_entry = {
#         "regressor_id": regressor_id,
#         "n_examples_trn": n_examples_trn,
#         "n_features_trn": n_features_trn,
#         "n_examples_val": n_examples_val,
#         "n_features_val": n_features_val,
#         "regressor_type": reg_type,
#         "regressor_kwargs": reg_kwargs,
#         "regressor_metrics": reg_metrics,
#         "timestamp": utils.get_timestamp(),
#         "ptblop_version": v_ptblop,
#         "ptblopgen_version": v_ptblopben,
#     }
#     update_db(regressor_db_path, db_entry)
#     return scoring_fn


def bp_configs_scores_sample(bp_configs, bp_config_scores, n: int, rng):
    if n < 0 or len(bp_configs) < n:
        return bp_configs, bp_config_scores
    else:
        indices = rng.sample(range(len(bp_configs)), k=n)
        bp_configs_new = [bp_configs[i] for i in indices]
        bp_config_scores_new = [bp_config_scores[i] for i in indices]
        return bp_configs_new, bp_config_scores_new


def gen_sample_configs(
    trn_schedule: list[configurator.TrnDataBlockConfig],
) -> collections.abc.Generator[tuple[int, int, int, int], None, None]:

    for t in trn_schedule:
        yield t.n_trn_onel, t.n_trn_rand, t.n_trn_actl, t.n_trn_parf

    t = trn_schedule[-1]

    while True:
        yield t.n_trn_onel, t.n_trn_rand, t.n_trn_actl, t.n_trn_parf


# def sample_one_layer_bp_configs(
#     *,
#     n,
#     bp_config_id_prefix,
#     model,
#     device,
#     evaluator_fn,
#     bp_config_db_path,
#     stop_path,
#     rng,
#     processed_bp_config_signatures,
# ):
#     logger.info(f"Started one layer sampling {n=}")
#     bp_config_unpruned = ptblop.get_unpruned_bp_config(model)

#     bp_configs, bp_config_scores = make_one_layer_bp_configs(bp_config_unpruned)

#     bp_configs, bp_config_scores = bp_configs_scores_sample(
#         bp_configs, bp_config_scores, n, rng
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
#     logger.info(f"Finished one layer sampling {n=}")


# def sample_random_bp_configs(
#     *,
#     max_num_changes,
#     n,
#     bp_config_id_prefix,
#     model,
#     device,
#     evaluator_fn,
#     bp_config_db_path,
#     stop_path,
#     rng,
#     processed_bp_config_signatures,
# ) -> None:
#     logger.info(f"Started random sampling {n=}")
#     bp_config_unpruned = ptblop.get_unpruned_bp_config(model)
#     bp_configs, bp_config_scores = make_random_bp_configs(
#         bp_config_unpruned=bp_config_unpruned,
#         n=n,
#         rng=rng,
#         min_num_changes=2,
#         max_num_changes=max_num_changes,
#         max_random_config_trials=MAX_RANDOM_CONFIG_TRIALS,
#         processed_bp_bconfig_signatures=processed_bp_config_signatures,
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
#     logger.info(f"Finihsed random sampling {n=}")


def conv_quality_estimator_to_scoring_fn(quality_estimator):

    def __scoring_fn(bp_config):
        X = estimator_helpers.get_quality_features([bp_config])
        _, ypred_min, ypred_max = quality_estimator.predict_with_bounds(X)
        return np.abs(ypred_max - ypred_min).item()

    return __scoring_fn


# def sample_active_learning_bp_configs(
#     *,
#     max_num_changes,
#     quality_estimator,
#     num_scored_candidates,
#     n,
#     bp_config_id_prefix,
#     model,
#     device,
#     evaluator_fn,
#     bp_config_db_path,
#     stop_path,
#     rng,
#     processed_bp_config_signatures,
# ):
#     logger.info(f"Started active learning sampling {n=}")
#     bp_config_unpruned = ptblop.get_unpruned_bp_config(model)
#     scoring_fn = conv_quality_estimator_to_scoring_fn(quality_estimator)
#     bp_configs, bp_config_scores = make_random_bp_configs_with_scoring(
#         bp_config_unpruned=bp_config_unpruned,
#         n=n,
#         rng=rng,
#         min_num_changes=2,
#         max_num_changes=max_num_changes,
#         processed_bp_bconfig_signatures=processed_bp_config_signatures,
#         max_random_config_trials=MAX_RANDOM_CONFIG_TRIALS,
#         scoring_fn=scoring_fn,
#         num_scored_candidates=num_scored_candidates,
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
#     logger.info(f"Finished active learning sampling {n=}")


# def make_pareto_bp_configs(
#     pareto_front_path,
#     n,
#     rng,
#     quality_metric,
#     min_quality,
#     min_num_changes,
#     max_num_changes,
#     processed_bp_bconfig_signatures,
# ):
#     quality_str = f"{quality_metric}_pred"

#     def __is_not_proc(d):
#         return (
#             get_bp_config_signature(d["bp_config"])
#             not in processed_bp_bconfig_signatures
#         )

#     with open(pareto_front_path, "rt") as f:
#         pf_data_raw = [json.loads(line) for line in f]

#     pf_data_filtered1 = []

#     # Filter configs satisying min/max_num_changes + min_quality criteria
#     for d in pf_data_raw:
#         num_changes = 2 * d["n"] - d["n_attention"] - d["n_mlp"]
#         logger.info(f"{num_changes=} {min_num_changes=} {max_num_changes=}")
#         if (
#             d[quality_str] > min_quality
#             and num_changes >= min_num_changes
#             and num_changes <= max_num_changes
#         ):
#             pf_data_filtered1.append(d)
#     logger.info(
#         f"Filtered  {len(pf_data_filtered1)} ouf {len(pf_data_raw)} "
#         f"bp_configs from {pareto_front_path}"
#     )

#     # Filter already processed signatures !!!!
#     pf_data_filtered2 = [d for d in pf_data_filtered1 if __is_not_proc(d)]
#     if len(pf_data_filtered1) < len(pf_data_filtered2):
#         logger.info(
#             f"Keept {len(pf_data_filtered2)} unprocessed configs "
#             f"out of {len(pf_data_filtered1)}"
#         )

#     n_tot = len(pf_data_filtered2)

#     if n_tot <= n:
#         bp_configs = [d["bp_config"] for d in pf_data_filtered2]
#         bp_config_scores = [d[f"{quality_metric}_pred"] for d in pf_data_filtered2]
#         return bp_configs, bp_config_scores
#     else:
#         # Sampling indices to keep the order of the configs
#         indices = list(range(len(pf_data_filtered2)))
#         indices_selected = sorted(rng.sample(indices, k=2))
#         bp_configs = [pf_data_filtered2[i]["bp_config"] for i in indices_selected]
#         bp_config_scores = [pf_data_filtered2[i][quality_str] for i in indices_selected]
#         return bp_configs, bp_config_scores


# def sample_pareto_front_bp_configs(
#     *,
#     max_num_changes: int,
#     pareto_front_path: pathlib.Path,
#     quality_metric: str,
#     min_quality: float,
#     n,
#     bp_config_id_prefix,
#     model,
#     device,
#     evaluator_fn,
#     bp_config_db_path: pathlib.Path,
#     stop_path: pathlib.Path,
#     rng,
#     processed_bp_config_signatures,
# ):
#     logger.info(f"Started Pareto front sampling {n=}")
#     bp_configs, bp_config_scores = make_pareto_bp_configs(
#         pareto_front_path=pareto_front_path,
#         n=n,
#         rng=rng,
#         quality_metric=quality_metric,
#         min_quality=min_quality,
#         min_num_changes=2,
#         max_num_changes=max_num_changes,
#         processed_bp_bconfig_signatures=processed_bp_config_signatures,
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
#     logger.info(f"Finished pareto front sampling {n=}")


def make_old_bp_config_spec_generator(bp_config_db_path: pathlib.Path):

    def __read_ids_types_signatures(bp_config_db_path):

        if bp_config_db_path.exists():
            res = []
            with open(bp_config_db_path, "rt") as f:
                for line in f:
                    d = json.loads(line)
                    signature = get_bp_config_signature(d["bp_config"])
                    res_entry = d["id"], d["type"], signature
                    res.append(res_entry)
            return res
        else:
            return []

    def __gen_ids_types_singatures(ids_types_singatures):
        for id_type_singature in ids_types_singatures:
            yield id_type_singature
        while True:
            yield None, None, None

    ids_types_singatures = __read_ids_types_signatures(bp_config_db_path)
    restart = len(ids_types_singatures) > 0
    return __gen_ids_types_singatures(ids_types_singatures), restart


def make_iteration_bp_config_spec_generator(i, n_val_rand, trn_schedule, max_onel):

    def __get_num_onel(n, max_onel):
        if n == -1 or n > max_onel:
            return max_onel
        else:
            return n

    def __get_trn_iter_config(trn_schedule, i_trn):
        # Two times minus one -
        # First - iter are numbered 1..,
        # Second - iter one is for validation
        # Hence training is 2..
        index = i_trn - 1
        if index < len(trn_schedule):
            return trn_schedule[index]
        else:
            return trn_schedule[-1]

    if i == 1:
        # i == 1 is validation iter !
        for j in range(1, n_val_rand + 1):
            yield "rand", f"val.rand.0001.{j:04d}"
    else:
        i_trn = i - 1
        dblc = __get_trn_iter_config(trn_schedule, i_trn)

        for j in range(1, dblc.n_trn_zerl + 1):
            yield "zerl", f"trn.zerl.{i_trn:04d}.{j:04d}"

        for j in range(1, __get_num_onel(dblc.n_trn_onel, max_onel) + 1):
            yield "onel", f"trn.onel.{i_trn:04d}.{j:04d}"

        for j in range(1, dblc.n_trn_rand + 1):
            yield "rand", f"trn.rand.{i_trn:04d}.{j:04d}"

        for j in range(1, dblc.n_trn_actl + 1):
            yield "actl", f"trn.actl.{i_trn:04d}.{j:04d}"

        for j in range(1, dblc.n_trn_actl + 1):
            yield "parf", f"trn.parf.{i_trn:04d}.{j:04d}"


def iter_range(num_iter):
    if num_iter < 0:
        i = 1
        while True:
            yield i
            i += 1
    else:
        yield from range(1, num_iter + 1)


def sample_bp_config(
    bp_config_id,
    bp_config_type,
    data_iter,
    rand_config_env,
    actl_config_env,
    parf_config_env,
    comm_config_env,
) -> int:
    pbpcs = comm_config_env["processed_bp_config_signatures"]
    model = comm_config_env["model"]
    bp_config_unpruned = ptblop.get_unpruned_bp_config(model)
    if bp_config_type == "rand":
        bp_config, bp_config_score = make_random_bp_config(
            bp_config_unpruned=bp_config_unpruned,
            rng=comm_config_env["rng"],
            min_num_changes=2,
            max_num_changes=rand_config_env["max_num_changes"],
            max_random_config_trials=MAX_RANDOM_CONFIG_TRIALS,
            processed_bp_bconfig_signatures=pbpcs,
        )
    elif bp_config_type == "actl":
        bp_config, bp_config_score = None, None
        msg = f"{bp_config_type=} for {bp_config_id=} not implemented"
        logger.warning(msg)
    elif bp_config_type == "parf":
        bp_config, bp_config_score = None, None
        msg = f"{bp_config_type=} for {bp_config_id=} not implemented"
        logger.warning(msg)
    elif bp_config_type == "onel":
        bp_config, bp_config_score = None, None
        msg = f"{bp_config_type=} for {bp_config_id=} not implemented"
        logger.warning(msg)
    elif bp_config_type == "zerl":
        bp_config, bp_config_score = None, None
        msg = f"{bp_config_type=} for {bp_config_id=} not implemented"
        logger.warning(msg)
    else:
        raise ValueError(f"Unknown {bp_config_type=}")

    if bp_config is not None and bp_config_score is not None:
        exit_code = process_bp_config(
            bp_config_id=bp_config_id,
            bp_config_type=bp_config_type,
            bp_config=bp_config,
            bp_config_score=bp_config_score,
            data_iter=data_iter,
            bp_config_db_path=comm_config_env["bp_config_db_path"],
            model=model,
            device=comm_config_env["device"],
            evaluator_fn=comm_config_env["evaluator_fn"],
            processed_bp_config_signatures=pbpcs,
            stop_path=comm_config_env["stop_path"],
        )
        return exit_code
    else:
        msg = f"Generation for {bp_config_id=}, {bp_config_type=} unsuccessful"
        logger.warning(msg)
        return 0


def main_modelgen(config: dict[str, Any], output_path: pathlib.Path) -> None:

    config_sampler = configurator.SamplerConfig(**config["sampler"])
    config_pareto_optimization = configurator.ParetoOptimizationConfig(
        **config["pareto_optimization"]
    )
    bp_config_db_path = output_path / BP_CONFIG_DB_FNAME
    quality_estimators_db_path = output_path / QUALITY_ESTIMATOR_DB_FNAME
    cost_estimators_db_path = output_path / COST_ESTIMATOR_DB_FNAME
    quality_estimators_report_path = output_path / QUALITY_ESTIMATOR_REPORT_DIR
    stop_path = output_path / STOP_FNAME

    quality_estimators_report_path.mkdir(exist_ok=True, parents=True)
    bp_config_db_spec_gen, restart = make_old_bp_config_spec_generator(
        bp_config_db_path
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, evaluator_fn = builders.make_model_and_evaluator(
        config["model"], config["evaluator"], device
    )

    bp_config_unpruned = ptblop.get_unpruned_bp_config(model)
    ptblop.apply_bp_config_in_place(model, {})

    if restart:
        rng = random.Random(None)
        seed = config_sampler.random_bp_config_rng_seed
        logger.info(f"This is a restart, None is used instead of {seed=}")
    else:
        rng = random.Random(config_sampler.random_bp_config_rng_seed)

    # Two because each block is two possible changes (mlp and attention)
    max_num_changes = round(
        2 * config_sampler.max_num_changes_factor * len(bp_config_unpruned)
    )
    logger.info(
        f"{max_num_changes=} num_blocks={len(bp_config_unpruned)} "
        f"max_num_changes_factor={config_sampler.max_num_changes_factor}"
    )

    processed_bp_config_signatures = set()
    cost_estimator, quality_estimator, pareto_front_path = None, None, None

    comm_config_env = {
        "bp_config_db_path": bp_config_db_path,
        "stop_path": stop_path,
        "device": device,
        "model": model,
        "evaluator_fn": evaluator_fn,
        "rng": rng,
        "processed_bp_config_signatures": processed_bp_config_signatures,
    }
    rand_config_env = {"max_num_changes": max_num_changes}
    actl_config_env = {
        "quality_estimator": None,
        "num_scored_candidates": config_sampler.actl_num_scored_candidates,
    }
    parf_config_env = {
        "max_num_changes": max_num_changes,
        "pareto_front_path": None,
        "quality_metric": config_sampler.quality_evaluator_metric,
        "min_quality": config_sampler.parf_min_quality_evaluator_metric,
    }

    for data_iter in iter_range(config_sampler.n_data_iter):

        iter_spec_gen = make_iteration_bp_config_spec_generator(
            data_iter,
            config_sampler.n_val_rand,
            config_sampler.trn_data_iter_configs,
            len(bp_config_unpruned),
        )

        for bp_config_type, bp_config_id in iter_spec_gen:
            bp_config_id_cfg, bp_config_type_cfg, bp_config_signature = next(
                bp_config_db_spec_gen
            )
            if (
                bp_config_id_cfg == bp_config_id
                and bp_config_type_cfg == bp_config_type
            ):
                processed_bp_config_signatures.add(bp_config_signature)
            elif bp_config_id_cfg is None and bp_config_type_cfg is None:
                if bp_config_type == "actl":
                    if quality_estimator is None:
                        quality_estimator = None  # TODO
                    actl_config_env["quality_estiator"] = quality_estimator
                parf_config_env["pareto_front_path"] = pareto_front_path
                sample_bp_config(
                    bp_config_id=bp_config_id,
                    bp_config_type=bp_config_type,
                    data_iter=data_iter,
                    rand_config_env=rand_config_env,
                    actl_config_env=actl_config_env,
                    parf_config_env=parf_config_env,
                    comm_config_env=comm_config_env,
                )
            else:
                msg = f"Schedules differ: {bp_config_id=}, {bp_config_type=} "
                msg += f"vs {bp_config_id_cfg=}, {bp_config_type_cfg=}"
                raise ValueError(msg)

        # data_iter == 1 -> validation, we cannot compute predictors + pareto fronts

        if data_iter != 1:
            pf_basename = PARETO_FRONT_FNAME_TEMPLATE % data_iter
            pareto_front_path = output_path / PARETO_FRONT_DIR / pf_basename

            if pareto_front_path.exists():
                msg = "Pareto front {pareto_front_path} exists, skipping generation"
                logger.info(msg)
            else:
                # Train predictors

                if cost_estimator is None:
                    cost_estimator_id = "estimator_quality_%04d" % data_iter
                    cost_estimator, cost_estimator_metrics = (
                        estimator_helpers.train_param_estimator(bp_config_db_path)
                    )
                    ceid = {"estimator_id": cost_estimator_id}
                    update_db(cost_estimators_db_path, ceid | cost_estimator_metrics)

                quality_estimator_id = "estimator_quality_%04d" % data_iter
                quality_estimator, quality_estimator_metrics = (
                    estimator_helpers.train_quality_estimator(
                        bp_config_db_path=bp_config_db_path,
                        quality_metric=config_sampler.quality_evaluator_metric,
                        quality_estimator_id=quality_estimator_id,
                        quality_estimator_report_path=quality_estimators_report_path,
                    )
                )
                qeid = {"estimator_id": quality_estimator_id}
                update_db(quality_estimators_db_path, qeid | quality_estimator_metrics)
                n_features = quality_estimator_metrics["n_features_trn"]

                # Generate Pareto front

                pareto_optimization.find_pareto_front(
                    quality_estimator=quality_estimator,
                    quality_metric_name=config_sampler.quality_evaluator_metric,
                    cost_estimator=cost_estimator,
                    n_features=n_features,
                    bp_config_unpruned=bp_config_unpruned,
                    pareto_path=pareto_front_path,
                    config_pareto_optimization=config_pareto_optimization,
                )

    # # Validation dataset

    # sample_random_bp_configs(
    #     n=config_sampler.n_val_rand,
    #     bp_config_id_prefix="val.rndl.0001.",
    #     max_num_changes=max_num_changes,
    #     **fixed_kwargs,
    # )

    # # Training dataset - unpruned model

    # process_bp_configs(
    #     bp_configs=[bp_config_unpruned],
    #     bp_config_scores=[-1],
    #     bp_config_id_prefix="trn.zerl.0001.",
    #     bp_config_db_path=bp_config_db_path,
    #     model=model,
    #     evaluator_fn=evaluator_fn,
    #     device=device,
    #     processed_bp_config_signatures=processed_bp_config_signatures,
    #     stop_path=stop_path,
    # )

    # # Training dataset - iterations as defined in config

    # quality_estimator, cost_estimator, pareto_front_path = None, None, None

    # for i, n_tuple in enumerate(
    #     gen_sample_configs(config_sampler.trn_schedule), start=1
    # ):
    #     n_onel, n_rand, n_actl, n_parf = n_tuple

    #     # Sample - single layer configs

    #     sample_one_layer_bp_configs(
    #         n=n_onel,
    #         bp_config_id_prefix=f"trn.onel.{i:04d}.",
    #         **fixed_kwargs,
    #     )

    #     # Sample - random configs

    #     sample_random_bp_configs(
    #         n=n_rand,
    #         max_num_changes=max_num_changes,
    #         bp_config_id_prefix=f"trn.rand.{i:04d}.",
    #         **fixed_kwargs,
    #     )

    #     # Sample - active learning

    #     if quality_estimator is not None:
    #         sample_active_learning_bp_configs(
    #             n=n_actl,
    #             quality_estimator=quality_estimator,
    #             max_num_changes=max_num_changes,
    #             num_scored_candidates=config_sampler.actl_num_scored_candidates,
    #             bp_config_id_prefix=f"trn.actl.{i:04d}.",
    #             **fixed_kwargs,
    #         )
    #     else:
    #         msg = "No quality predictor, but active learning sampling requested"
    #         logger.warning(msg)

    #     # Sample - Pareto front

    #     if pareto_front_path is not None:
    #         sample_pareto_front_bp_configs(
    #             n=n_parf,
    #             max_num_changes=max_num_changes,
    #             pareto_front_path=pareto_front_path,
    #             quality_metric=config_sampler.quality_evaluator_metric,
    #             min_quality=config_sampler.parf_min_quality_evaluator_metric,
    #             bp_config_id_prefix=f"trn.parf.{i:04d}.",
    #             **fixed_kwargs,
    #         )
    #     else:
    #         logger.warning("No Pareto fron, but Pareto front sampling requested")

    #     # Train predictors

    #     if cost_estimator is None:
    #         cost_estimator_id = "estimator_quality_%04d" % i
    #         cost_estimator, cost_estimator_metrics = (
    #             estimator_helpers.train_param_estimator(bp_config_db_path)
    #         )
    #         ceid = {"estimator_id": cost_estimator_id}
    #         update_db(cost_estimators_db_path, ceid | cost_estimator_metrics)

    #     quality_estimator_id = "estimator_quality_%04d" % i
    #     quality_estimator, quality_estimator_metrics = (
    #         estimator_helpers.train_quality_estimator(
    #             bp_config_db_path=bp_config_db_path,
    #             quality_metric=config_sampler.quality_evaluator_metric,
    #             quality_estimator_id=quality_estimator_id,
    #             quality_estimator_report_path=quality_estimators_report_path,
    #         )
    #     )
    #     qeid = {"estimator_id": quality_estimator_id}
    #     update_db(quality_estimators_db_path, qeid | quality_estimator_metrics)
    #     n_features = quality_estimator_metrics["n_features_trn"]

    #     # Generate pareto front

    #     pareto_front_path = (
    #         output_path / PARETO_FRONT_DIR / (PARETO_FRONT_FNAME_TEMPLATE % i)
    #     )
    #     pareto_optimization.find_pareto_front(
    #         quality_estimator=quality_estimator,
    #         quality_metric_name=config_sampler.quality_evaluator_metric,
    #         cost_estimator=cost_estimator,
    #         n_features=n_features,
    #         bp_config_unpruned=bp_config_unpruned,
    #         pareto_path=pareto_front_path,
    #         config_pareto_optimization=config_pareto_optimization,
    #     )
