import copy
import json
import logging
import pathlib
import random
from typing import Any

import numpy as np
import ptblop
import torch

from .. import _version, builders, regressors
from . import configurator, regressor_helpers

BPCONFIG_DB_FNAME = "bpconfigs.json"
REGRESSORS_DB_FNAME = "regressors.json"
STOP_FNAME = "STOP"

MAX_RANDOM_CONFIG_TRIALS = 20

logger = logging.getLogger(__name__)


def update_db(db_path, db_entry, mode="append"):
    if mode == "append":
        flag = "at"
    elif mode == "reset":
        flag = "wt"
    else:
        raise ValueError(f"Unknown mode {mode}")

    with open(db_path, flag) as f:
        f.write(json.dumps(db_entry) + "\n")


# Helpers


def get_num_params(m: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


# Model - benchmarking


def process_single_bp_config(*, model, device, bp_config_data, evaluator_fn):
    bp_config, bp_config_id, bp_config_score = bp_config_data
    res = {"id": bp_config_id}
    res["n_attention"] = ptblop.get_num_attention_blocks(model)
    res["n_mlp"] = ptblop.get_num_mlp_blocks(model)
    res["n"] = ptblop.get_num_prunable_blocks(model)
    res["mparams"] = ptblop.get_num_params(model) / 1.0e6
    ptblop.apply_bp_config_in_place(model, bp_config, set_unused_layers_to_none=False)
    res |= evaluator_fn(model, device)
    res["bpconfig_score"] = bp_config_score
    res["bpconfig"] = bp_config
    res["ptblop"] = _version.__version__
    return res


# Bpconfigs - helper functions


def genereate_bpconfig_changes(bpconfog_unpruned):
    config_changes = []

    for k, v in bpconfog_unpruned.items():
        for k1 in v:
            config_changes.append({k: {k1: False}})
    return config_changes


def get_bpconfig_signature(bpconfig):
    singature_strs = []

    for v in bpconfig.values():
        v_signature_str = str(int(not v["use_attention"])) + str(int(not v["use_mlp"]))
        singature_strs.append(v_signature_str)
    signature_str = "".join(singature_strs)
    return int(signature_str, 2)


def apply_bpconfig_changes(bpconfig_unpruned, bpconfig_changes):
    blockprune_cfg = copy.deepcopy(bpconfig_unpruned)

    for c in bpconfig_changes:
        for k, v in c.items():
            blockprune_cfg[k] |= v

    return blockprune_cfg


def are_all_bp_configs_processed(bpconfigs, bpconfig_id_fmt, bpconfig_db_path):
    if bpconfig_db_path.exists():
        bpconfig_ids = {(bpconfig_id_fmt % i) for i, _ in enumerate(bpconfigs, start=1)}
        processed_bpconfigs_ids = set()

        with open(bpconfig_db_path, "rt") as f:
            for line in f:
                d = json.loads(line)
                processed_bpconfigs_ids.add(d["id"])

        n1 = len(bpconfig_ids)
        n2 = len(bpconfig_ids & processed_bpconfigs_ids)
        logger.info(f"{bpconfig_id_fmt} {n1=} {n2=}")
        return n1 == n2
    else:
        return False


def process_bp_configs(
    *,
    bp_configs,
    bp_config_scores,
    bpconfig_id_prefix,
    bpconfig_db_path,
    model,
    device,
    evaluator_fn,
    processed_bpconfig_signatures,
):
    assert len(bp_configs) == len(bp_config_scores)
    bpconfig_id_fmt = f"{bpconfig_id_prefix}%04d"

    if are_all_bp_configs_processed(bp_configs, bpconfig_id_fmt, bpconfig_db_path):
        logger.info(f"All configs from batch {bpconfig_id_prefix} already processed")
    else:
        bpconfigs_and_scores = zip(bp_configs, bp_config_scores)
        for i, (bpconfig, bpconfig_score) in enumerate(bpconfigs_and_scores, start=1):
            bpconfig_signature = get_bpconfig_signature(bpconfig)
            if bpconfig_signature in processed_bpconfig_signatures:
                logger.warning(f"Model already processed {bpconfig_signature=}")
            else:
                bpconfig_id = bpconfig_id_fmt % i
                bp_config_data = (bpconfig, bpconfig_id, bpconfig_score)
                res = process_single_bp_config(
                    bp_config_data=bp_config_data,
                    model=model,
                    device=device,
                    evaluator_fn=evaluator_fn,
                )
                assert bpconfig_signature not in processed_bpconfig_signatures
                processed_bpconfig_signatures.add(bpconfig_signature)
                update_db(bpconfig_db_path, res)


def read_processed_bp_config_signatures(db_path):
    signatures = set()

    with open(db_path, "rt") as f:
        for line in f:
            d = json.loads(line)
            bpconfig = d["bpconfig"]
            sig = get_bpconfig_signature(bpconfig)
            signatures.add(sig)
    logger.info(f"Read {len(signatures)} configurations and stored their signatures")
    return signatures


# Bpconfigs - generation - one layer


def make_one_layer_bp_configs(bpconfig_unpruned):
    cfg_changes_all = genereate_bpconfig_changes(bpconfig_unpruned)
    res = []

    for cfg_change in cfg_changes_all:
        bpconfig = copy.deepcopy(bpconfig_unpruned)
        bpconfig = apply_bpconfig_changes(bpconfig, [cfg_change])
        res.append(bpconfig)

    return res, [-1.0 for r in res]


# Bpconfigs - generation - random


def _make_random_bp_config(
    *,
    rng,
    bpconfig_unpruned,
    num_changes,
    processed_bp_config_signatures,
    cfg_changes_all=None,
    max_random_config_trials,
):
    # TODO Delete this line
    if cfg_changes_all is None:
        cfg_changes_all = genereate_bpconfig_changes(bpconfig_unpruned)

    bpconfig, bpconfig_signature = None, None

    for j in range(1, max_random_config_trials + 1):
        cfg_changes = rng.sample(cfg_changes_all, k=num_changes)
        bpconfig = apply_bpconfig_changes(bpconfig_unpruned, cfg_changes)
        bpconfig_signature = get_bpconfig_signature(bpconfig)

        if bpconfig_signature not in processed_bp_config_signatures:
            break
        else:
            msg = f"Try {j=}, drew signature={bpconfig_signature}"
            msg += " that is already processed, repeating"
            logger.warning(msg)

    return bpconfig, bpconfig_signature


def make_random_bp_configs(
    *,
    bpconfig_unpruned,
    num_configs,
    rng,
    min_num_changes,
    max_num_changes,
    max_random_config_trials,
    processed_bpbconfig_signatures,
):
    processed_bpbconfig_signatures_all = copy.deepcopy(processed_bpbconfig_signatures)
    cfg_changes_all = genereate_bpconfig_changes(bpconfig_unpruned)

    res = []
    while len(res) < num_configs:

        num_changes = rng.randint(min_num_changes, max_num_changes)
        bpconfig, bpconfig_signature = _make_random_bp_config(
            rng=rng,
            bpconfig_unpruned=bpconfig_unpruned,
            num_changes=num_changes,
            processed_bp_config_signatures=processed_bpbconfig_signatures_all,
            cfg_changes_all=cfg_changes_all,
            max_random_config_trials=max_random_config_trials,
        )
        if bpconfig is not None:
            res.append(bpconfig)
            processed_bpbconfig_signatures_all.add(bpconfig_signature)
    res_scores = [-1.0 for r in res]  # For random configs scores are meaningless
    return res, res_scores


# Bpconfigs - generation - with scoring


def make_random_bpconfigs_with_scoring(
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
    cfg_changes_all = genereate_bpconfig_changes(bp_config_unpruned)

    final_bpconfigs = []
    final_scores = []
    while len(final_bpconfigs) < num_configs:

        num_changes = rng.randint(min_num_changes, max_num_changes)

        # Generate candidates

        candidate_bpconfigs = []
        candidate_signatures = set()
        for _ in range(num_scored_candidates):
            tmp_singnatures = processed_bpbconfig_signatures_all | candidate_signatures
            bpconfig, bpconfig_signature = _make_random_bp_config(
                rng=rng,
                bpconfig_unpruned=bp_config_unpruned,
                num_changes=num_changes,
                processed_bp_config_signatures=tmp_singnatures,
                cfg_changes_all=cfg_changes_all,
            )
            if bpconfig is not None:
                candidate_bpconfigs.append(bpconfig)
                candidate_signatures.add(bpconfig_signature)

        # Select highest scored candidate

        if len(candidate_bpconfigs) > 0:
            n_c = len(candidate_bpconfigs)
            logger.info(f"Generated {n_c} config candidates, selecting one config")
            scores = np.array([scoring_fn(bpc) for bpc in candidate_bpconfigs])
            imax = np.argmax(scores)
            bpconfig = candidate_bpconfigs[imax]
            max_score = scores[imax]
            min_score = np.min(scores)
            mean_score = np.mean(scores)
            assert max_score == scoring_fn(bpconfig)

            logger.info(f"Score stats: {max_score=} {min_score=} {mean_score=}")
            # scores = [scoring_fn(bpc) for bpc in candidate_bpconfigs]
            # scores.sort(reverse=True)

            # for i, r in enumerate(scores, start=1):
            #     logger.info(f"Scoring {i} - {r}")

            bpconfig_signature = get_bpconfig_signature(bpconfig)
            final_bpconfigs.append(bpconfig)
            final_scores.append(max_score)
            processed_bpbconfig_signatures_all.add(bpconfig_signature)
        else:
            logger.info(f"Failed to generate any configs for {num_changes=}")

    return final_bpconfigs, final_scores


# Bpconfigs - predictor based scoring


def conv_regressor_to_scoring_fn(reg):

    def __scoring_fn(bp_config):
        X = regressor_helpers.get_quality_features([bp_config])
        _, ypred_min, ypred_max = reg.predict_with_bounds(X)
        return np.abs(ypred_max - ypred_min).item()

    return __scoring_fn


def make_scoring_fn(regressor_id, bpconfig_db_path, regressor_db_path):
    data_trn, data_val = regressor_helpers.read_data(bpconfig_db_path)

    X_trn = regressor_helpers.get_quality_features([d["bpconfig"] for d in data_trn])
    y_trn = regressor_helpers.get_target(data_trn, "arc_challenge_acc")
    logger.info(f"{X_trn.shape=} {X_trn.dtype=} {y_trn.shape=} {y_trn.dtype=}")

    X_val = regressor_helpers.get_quality_features([d["bpconfig"] for d in data_val])
    y_val = regressor_helpers.get_target(data_val, "arc_challenge_acc")
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

    reg = regressors.QuantileGradientBoostingBoundsRegressor(**reg_kwargs)
    reg.fit(X_trn, y_trn)
    reg_metrics = regressor_helpers.evaluate_bounds_regressor(
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


# def make_seed_dataset(
#     *,
#     model_data,
#     device,
#     bpconfig_unpruned,
#     rng,
#     bpconfig_db_path,
#     processed_bpconfig_signatures,
# ):
#     # VAL - random configs

#     for i_val in range(1, 5):
#         bpconfigs, bpconfig_scores = make_random_bpconfigs(
#             bpconfig_unpruned=bpconfig_unpruned,
#             num_configs=N_VAL,
#             rng=rng,
#             min_num_changes=2,
#             max_num_changes=32,
#             processed_bpbconfig_signatures=processed_bpconfig_signatures,
#         )

#         process_bpconfigs(
#             bpconfigs=bpconfigs,
#             bpconfig_scores=bpconfig_scores,
#             bpconfig_id_prefix=f"val.rndl.{i_val:03d}.",
#             bpconfig_db_path=bpconfig_db_path,
#             model_data=model_data,
#             device=device,
#             processed_bpconfig_signatures=processed_bpconfig_signatures,
#         )

#     # TRN - Unpruned config

#     process_bpconfigs(
#         bpconfigs=[bpconfig_unpruned],
#         bpconfig_scores=[-1.0],
#         bpconfig_id_prefix="trn.zerl.001.",
#         bpconfig_db_path=bpconfig_db_path,
#         model_data=model_data,
#         device=device,
#         processed_bpconfig_signatures=processed_bpconfig_signatures,
#     )

#     # TRN - one layer configs

#     bpconfigs, bpconfig_scores = make_one_layer_bpconfigs(bpconfig_unpruned)

#     process_bpconfigs(
#         bpconfigs=bpconfigs[:N_TRN_LAYER_CONFIGS],
#         bpconfig_scores=bpconfig_scores[:N_TRN_LAYER_CONFIGS],
#         bpconfig_id_prefix="trn.onel.001.",
#         bpconfig_db_path=bpconfig_db_path,
#         model_data=model_data,
#         device=device,
#         processed_bpconfig_signatures=processed_bpconfig_signatures,
#     )

#     # TRN - random configs

#     for i_trn in range(1, 3):
#         bpconfigs, bpconfig_scores = make_random_bpconfigs(
#             bpconfig_unpruned=bpconfig_unpruned,
#             num_configs=N_TRN,
#             rng=rng,
#             min_num_changes=2,
#             max_num_changes=32,
#             processed_bpbconfig_signatures=processed_bpconfig_signatures,
#         )

#         process_bpconfigs(
#             bpconfigs=bpconfigs,
#             bpconfig_scores=bpconfig_scores,
#             bpconfig_id_prefix=f"trn.rndl.{i_trn:03d}.",
#             bpconfig_db_path=bpconfig_db_path,
#             model_data=model_data,
#             device=device,
#             processed_bpconfig_signatures=processed_bpconfig_signatures,
#         )


def main_sample_random(config: dict[str, Any], output_path: pathlib.Path) -> None:
    bpconfig_db_path = output_path / BPCONFIG_DB_FNAME
    stop_path = output_path / STOP_FNAME
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, evaluator_fn = builders.make_model_and_evaluator(
        config["model"], config["evaluator"], device
    )

    bp_config_unpruned = ptblop.get_unpruned_bp_config(model)

    config_sampler = configurator.SamplerConfig(**config["sampler"])
    max_num_changes = round(
        config_sampler.max_num_changes_factor * len(bp_config_unpruned)
    )
    rng = random.Random(config_sampler.random_bp_config_rng_seed)

    processed_bpconfig_signatures = set()

    # VAL DATASET
    bp_configs, bp_config_scores = make_random_bp_configs(
        bpconfig_unpruned=bp_config_unpruned,
        num_configs=config_sampler.n_val_rand,
        rng=rng,
        min_num_changes=2,
        max_num_changes=max_num_changes,
        max_random_config_trials=MAX_RANDOM_CONFIG_TRIALS,
        processed_bpbconfig_signatures=processed_bpconfig_signatures,
    )

    process_bp_configs(
        bp_configs=bp_configs,
        bp_config_scores=bp_config_scores,
        bpconfig_id_prefix="val.rndl.001.",
        bpconfig_db_path=bpconfig_db_path,
        model=model,
        evaluator_fn=evaluator_fn,
        device=device,
        processed_bpconfig_signatures=processed_bpconfig_signatures,
    )

    # make_seed_dataset(
    #     model_data=model_data,
    #     device=device,
    #     bpconfig_unpruned=bp_config_unpruned,
    #     rng=rng,
    #     bpconfig_db_path=bpconfig_db_path,
    #     processed_bpconfig_signatures=processed_bpconfig_signatures,
    # )

    # # TRN - random configs

    # i_trn = 4
    # while True:
    #     if stop_path.exists():
    #         logger.warning(f"{stop_path} found, exiting")
    #         break

    #     bpconfigs, bpconfig_scores = make_random_bpconfigs(
    #         bpconfig_unpruned=bp_config_unpruned,
    #         num_configs=N_TRN,
    #         rng=rng,
    #         min_num_changes=2,
    #         max_num_changes=32,
    #         processed_bpbconfig_signatures=processed_bpconfig_signatures,
    #     )

    #     process_bpconfigs(
    #         bpconfigs=bpconfigs,
    #         bpconfig_scores=bpconfig_scores,
    #         bpconfig_id_prefix=f"trn.rndl.{i_trn:03d}.",
    #         bpconfig_db_path=bpconfig_db_path,
    #         model_data=model_data,
    #         device=device,
    #         processed_bpconfig_signatures=processed_bpconfig_signatures,
    #     )
    #     i_trn += 1


# def read_pareto(fname):
#     data = []
#     with open(fname, "rt") as f:
#         for line in f:
#             data.append(json.loads(line))

#     bpconfigs = [d["bpconfig"] for d in data]
#     bpconfig_scores = [d["arc_challenge_acc_pred"] for d in data]
#     return bpconfigs, bpconfig_scores


# def main_eval_pareto(args):
#     bpconfig_db_path = args.output_path / BPCONFIG_DB_FNAME
#     pareto_db_path = args.output_path / "pareto.json"

#     model_name = MODEL_NAME
#     model_revision = MODEL_REVISION
#     model_dtype = MODEL_DTYPE
#     model_data = HFModelData(
#         name=model_name, revision=model_revision, dtype=model_dtype
#     )
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     processed_bpconfig_signatures = set()

#     bpconfigs, bpconfig_scores = read_pareto(pareto_db_path)

#     # # Experiment

#     # import sys
#     # import blockprunekit.modelgen.regressor_helpers as rh

#     # def _get_bpconfig_from_features(sample_bpconfig, features):
#     #     assert len(features) == 2 * len(sample_bpconfig)
#     #     res = {}

#     #     for i, k in enumerate(sample_bpconfig):
#     #         use_attention = float(features[2 * i]) > 0.5
#     #         use_mlp = float(features[2 * i + 1]) > 0.5
#     #         res[k] = {"use_attention": use_attention, "use_mlp": use_mlp}
#     #     return res

#     # bpconfig = bpconfigs[0]
#     # features = rh.get_quality_features([bpconfig])
#     # bpconfig_unpruned = get_bpconfig_unpruned(
#     #     model_name=model_name, model_revision=model_revision, model_dtype=model_dtype
#     # )
#     # bpconfig2 = _get_bpconfig_from_features(bpconfig_unpruned, features[0,:])
#     # print("bpconfig == bpconfig2", bpconfig == bpconfig2)
#     # sys.exit()

#     # # End experiment

#     process_bpconfigs(
#         bpconfigs=bpconfigs,
#         bpconfig_scores=bpconfig_scores,
#         bpconfig_id_prefix=f"prt.",
#         bpconfig_db_path=bpconfig_db_path,
#         model_data=model_data,
#         device=device,
#         processed_bpconfig_signatures=processed_bpconfig_signatures,
#     )

# def main_sample_active_learn(args):
#     bpconfig_db_path = args.output_path / BPCONFIG_DB_FNAME
#     stop_path = args.output_path / STOP_FNAME
#     regressor_db_path = args.output_path / REGRESSORS_DB_FNAME

#     model_name = MODEL_NAME
#     model_revision = MODEL_REVISION
#     model_dtype = MODEL_DTYPE
#     model_data = HFModelData(
#         name=model_name, revision=model_revision, dtype=model_dtype
#     )
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     bpconfig_unpruned = ptblop.get_()

#     if not args.output_path.exists():
#         args.output_path.mkdir(parents=True, exist_ok=False)

#     rng = random.Random(CONFIG_SAMPLER_SEED)
#     processed_bpconfig_signatures = set()

#     # VAL + TRN SEED DATASET

#     make_seed_dataset(
#         model_data=model_data,
#         device=device,
#         bpconfig_unpruned=bpconfig_unpruned,
#         rng=rng,
#         bpconfig_db_path=bpconfig_db_path,
#         processed_bpconfig_signatures=processed_bpconfig_signatures,
#     )

#     # TRN - ACTIVE LEARNING

#     i_trn_actl = 1

#     while True:
#         bpconfig_prefix = f"trn.actl.{i_trn_actl:03d}."
#         if stop_path.exists():
#             logger.warning(f"{stop_path} found, exiting")
#             break
#         scoring_fn = make_scoring_fn(
#             bpconfig_prefix, bpconfig_db_path, regressor_db_path
#         )
#         bpconfigs, bpconfig_scores = make_random_bpconfigs_with_scoring(
#             bpconfig_unpruned=bpconfig_unpruned,
#             num_configs=N_TRN,
#             rng=rng,
#             min_num_changes=2,
#             max_num_changes=MAX_NUM_CHANGES,
#             processed_bpbconfig_signatures=processed_bpconfig_signatures,
#             num_scored_candidates=NUM_RANKED_CANDIDATES,
#             scoring_fn=scoring_fn,
#         )
#         process_bpconfigs(
#             bpconfigs=bpconfigs,
#             bpconfig_scores=bpconfig_scores,
#             bpconfig_id_prefix=bpconfig_prefix,
#             bpconfig_db_path=bpconfig_db_path,
#             model_data=model_data,
#             device=device,
#             processed_bpconfig_signatures=processed_bpconfig_signatures,
#         )
#         i_trn_actl += 1
