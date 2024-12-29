import copy
import json
import logging
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import ptblop
import torch

from .. import builders, utils
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


@dataclass
class BPConfigProcsessingEnvironment:
    model: torch.nn.Module
    device: torch.device
    evaluator_fn: Any
    stop_path: pathlib.Path
    bp_config_db_path: pathlib.Path


class BPConfigGenerators:
    def __init__(self, zerl, onel, rand, parf, actl):
        self.zerl = zerl
        self.onel = onel
        self.rand = rand
        self.parf = parf
        self.actl = actl

    def get_gen(self, gen_name: str) -> Any:
        if hasattr(self, gen_name):
            return getattr(self, gen_name)
        else:
            raise ValueError("Unknow generator type {gen_name}")


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


def apply_bp_config_changes(bp_config_unpruned, bp_config_changes):
    bp_config = copy.deepcopy(bp_config_unpruned)

    for c in bp_config_changes:
        for k, v in c.items():
            bp_config[k] |= v

    return bp_config


def make_old_iter_generator(bp_config_db_path: pathlib.Path):

    def __read_ids_types_signatures(bp_config_db_path):

        if bp_config_db_path.exists():
            res = []
            with open(bp_config_db_path, "rt") as f:
                for line in f:
                    d = json.loads(line)
                    signature = utils.get_bp_config_signature(d["bp_config"])
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


def make_generator_for_data_iter(i, n_val_rand, trn_schedule, max_onel):

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

        for j in range(1, dblc.n_trn_parf + 1):
            yield "parf", f"trn.parf.{i_trn:04d}.{j:04d}"


def iter_range(num_iter):
    if num_iter < 0:
        i = 1
        while True:
            yield i
            i += 1
    else:
        yield from range(1, num_iter + 1)


# Generators


def make_mock_bp_config_generator():
    while True:
        yield None, None


def make_zerl_bp_config_generator(bp_config_unpruned):
    yield bp_config_unpruned, -1.0
    while True:
        yield None, None


def make_onel_bp_config_generator(bp_config_unpruned):

    # Generate all one point changes ans shuffle them

    cfg_changes_all = genereate_bp_config_changes(bp_config_unpruned)
    bp_configs = []

    for cfg_change in cfg_changes_all:
        bp_config = copy.deepcopy(bp_config_unpruned)
        bp_config = apply_bp_config_changes(bp_config, [cfg_change])
        bp_configs.append(bp_config)

    # This needs a fixed seed - so in case of restarting in the middle of
    # onel phase the bp config are generated in the same order
    rng = random.Random(712345)
    rng.shuffle(bp_configs)  # shuffle = in-place operation

    for bpc in bp_configs:
        yield bpc, -1.0

    while True:
        yield None, None


def make_parf_bp_config_generator(
    pareto_front_path,
    quality_metric,
    min_quality,
    min_num_changes,
    max_num_changes,
    rng,
    processed_bp_bconfig_signatures,
):
    # Read, filter and shuffle Pareto Front data
    quality_str = f"{quality_metric}_pred"

    def __is_not_proc(d):
        return (
            utils.get_bp_config_signature(d["bp_config"])
            not in processed_bp_bconfig_signatures
        )

    with open(pareto_front_path, "rt") as f:
        pf_data_raw = [json.loads(line) for line in f]

    pf_data_filtered1 = []

    # Filter configs satisying min/max_num_changes + min_quality criteria
    for d in pf_data_raw:
        num_changes = 2 * d["n"] - d["n_attention"] - d["n_mlp"]
        logger.info(f"{num_changes=} {min_num_changes=} {max_num_changes=}")
        if (
            d[quality_str] > min_quality
            and num_changes >= min_num_changes
            and num_changes <= max_num_changes
        ):
            pf_data_filtered1.append(d)
    logger.info(
        f"Filtered {len(pf_data_filtered1)} ouf {len(pf_data_raw)} "
        f"bp_configs from {pareto_front_path}"
    )

    # Filter already processed signatures !!!!
    pf_data_filtered2 = [d for d in pf_data_filtered1 if __is_not_proc(d)]
    if len(pf_data_filtered1) < len(pf_data_filtered2):
        logger.info(
            f"Keept {len(pf_data_filtered2)} unprocessed configs "
            f"out of {len(pf_data_filtered1)}"
        )

    # Sampling indices to keep the order of the configs
    indices = list(range(len(pf_data_filtered2)))
    rng.shuffle(indices)
    bp_configs = [pf_data_filtered2[i]["bp_config"] for i in indices]
    bp_config_scores = [pf_data_filtered2[i][quality_str] for i in indices]

    for bpc, bpc_score in zip(bp_configs, bp_config_scores):
        yield bpc, bpc_score

    while True:
        yield None, None


def make_rand_bp_config(
    *,
    cfg_changes_all,
    bp_config_unpruned,
    num_changes,
    max_random_config_trials,
    rng,
    processed_bp_config_signatures,
):

    bp_config, bp_config_signature = None, None

    for j in range(1, max_random_config_trials + 1):
        cfg_changes = rng.sample(cfg_changes_all, k=num_changes)
        bp_config = apply_bp_config_changes(bp_config_unpruned, cfg_changes)
        bp_config_signature = utils.get_bp_config_signature(bp_config)

        if bp_config_signature not in processed_bp_config_signatures:
            break
        else:
            msg = f"Try {j=}, {num_changes=},  drew signature={bp_config_signature}"
            msg += " that is already processed, repeating"
            logger.info(msg)
    if bp_config is not None:
        return bp_config, -1.0
    else:
        msg = "After {j=} tries, failed to generate bp_config for {num_changes=}"
        return None, None


def make_rand_bp_config_generator(
    *,
    bp_config_unpruned,
    min_num_changes,
    max_num_changes,
    max_random_config_trials,
    rng,
    processed_bp_config_signatures,
):
    cfg_changes_all = genereate_bp_config_changes(bp_config_unpruned)

    while True:
        num_changes = rng.randint(min_num_changes, max_num_changes)
        bp_config, bp_config_score = make_rand_bp_config(
            num_changes=num_changes,
            cfg_changes_all=cfg_changes_all,
            bp_config_unpruned=bp_config_unpruned,
            max_random_config_trials=max_random_config_trials,
            rng=rng,
            processed_bp_config_signatures=processed_bp_config_signatures,
        )
        if bp_config is not None:
            yield bp_config, bp_config_score


def conv_quality_estimator_to_scoring_fn(quality_estimator):

    def __scoring_fn(bp_configs):
        X = estimator_helpers.get_quality_features(bp_configs)
        _, ypred_min, ypred_max = quality_estimator.predict_with_bounds(X)
        return np.abs(ypred_max - ypred_min)

    return __scoring_fn


def make_actl_bp_config_generator(
    *,
    bp_config_unpruned,
    min_num_changes,
    max_num_changes,
    max_random_config_trials,
    rng,
    processed_bp_bconfig_signatures,
    num_scored_candidates,
    quality_estimator,
):
    cfg_changes_all = genereate_bp_config_changes(bp_config_unpruned)
    scoring_fn = conv_quality_estimator_to_scoring_fn(quality_estimator)

    while True:

        # Generate pool of candidates with fixed num_chages

        num_changes = rng.randint(min_num_changes, max_num_changes)
        processed_bpbconfig_signatures_all = copy.deepcopy(
            processed_bp_bconfig_signatures
        )

        candidate_bp_configs = []
        candidate_signatures = set()
        for _ in range(num_scored_candidates):
            tmp_singnatures = processed_bpbconfig_signatures_all | candidate_signatures
            max_bp_config, bp_config_signature = make_rand_bp_config(
                num_changes=num_changes,
                cfg_changes_all=cfg_changes_all,
                bp_config_unpruned=bp_config_unpruned,
                max_random_config_trials=max_random_config_trials,
                rng=rng,
                processed_bp_config_signatures=tmp_singnatures,
            )
            if max_bp_config is not None:
                candidate_bp_configs.append(max_bp_config)
                candidate_signatures.add(bp_config_signature)

        # Select highest scored candidate

        if len(candidate_bp_configs) > 0:
            n_c = len(candidate_bp_configs)
            logger.info(f"actl - {num_changes=}, generated {n_c=} config candidates")
            t1 = time.perf_counter()
            scores = scoring_fn(candidate_bp_configs)
            scoring_duration = time.perf_counter() - t1
            logger.info(f"actl - scoring candidates took {scoring_duration:.2f} sec.")
            imax = np.argmax(scores)
            max_bp_config = candidate_bp_configs[imax]
            max_score = float(scores[imax])
            min_score = float(np.min(scores))
            mean_score = float(np.mean(scores))
            assert max_score == scoring_fn([max_bp_config]).item()

            logger.info(f"actl - {max_score=:.4f} {min_score=:.4f} {mean_score=:.4f}")
            # scores = [scoring_fn(bpc) for bpc in candidate_bp_configs]
            # scores.sort(reverse=True)

            # for i, r in enumerate(scores, start=1):
            #     logger.info(f"Scoring {i} - {r}")
            yield max_bp_config, max_score
        else:
            logger.info(f"actl - {num_changes=}, failed to generate config candidates")


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
    # TODO Merge this with sample_and_process_bp_config

    if stop_path.exists():
        logger.warning(f"Stop file found {stop_path}, exiting...")
        return
    bp_config_signature = utils.get_bp_config_signature(bp_config)
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
        res["bp_config_signature"] = hex(bp_config_signature)[2:]
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


def sample_and_process_bp_config(
    bp_config_id,
    bp_config_type,
    data_iter,
    bp_config_gens,
    processing_env: BPConfigProcsessingEnvironment,
    processed_bp_config_signatures,
) -> int:
    # TODO: Merge this with process_bp_config

    bp_config, bp_config_score = next(bp_config_gens.get_gen(bp_config_type))
    if bp_config is not None and bp_config_score is not None:
        exit_code = process_bp_config(
            bp_config_id=bp_config_id,
            bp_config_type=bp_config_type,
            bp_config=bp_config,
            bp_config_score=bp_config_score,
            data_iter=data_iter,
            model=processing_env.model,
            device=processing_env.device,
            evaluator_fn=processing_env.evaluator_fn,
            bp_config_db_path=processing_env.bp_config_db_path,
            stop_path=processing_env.stop_path,
            processed_bp_config_signatures=processed_bp_config_signatures,
        )
        return exit_code
    else:
        msg = f"Generation for {bp_config_id=}, {bp_config_type=} unsuccessful"
        logger.warning(msg)
        return 0


def make_bp_config_processing_env(
    config: dict[str, Any], output_path: pathlib.Path
) -> BPConfigProcsessingEnvironment:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, evaluator_fn = builders.make_model_and_evaluator(
        config["model"], config["evaluator"], device
    )
    ptblop.apply_bp_config_in_place(model, {})

    bp_config_db_path = output_path / BP_CONFIG_DB_FNAME
    stop_path = output_path / STOP_FNAME

    return BPConfigProcsessingEnvironment(
        model=model,
        device=device,
        evaluator_fn=evaluator_fn,
        stop_path=stop_path,
        bp_config_db_path=bp_config_db_path,
    )


def make_bp_config_generators(
    *,
    bp_config_unpruned,
    min_num_changes,
    max_num_changes,
    processed_bp_config_singatures,
    rng,
) -> BPConfigGenerators:
    zerlg = make_zerl_bp_config_generator(bp_config_unpruned)
    onelg = make_onel_bp_config_generator(bp_config_unpruned)
    randg = make_rand_bp_config_generator(
        bp_config_unpruned=bp_config_unpruned,
        min_num_changes=min_num_changes,
        max_num_changes=max_num_changes,
        max_random_config_trials=MAX_RANDOM_CONFIG_TRIALS,
        rng=rng,
        processed_bp_config_signatures=processed_bp_config_singatures,
    )
    parfg = None
    actlg = None
    return BPConfigGenerators(
        zerl=zerlg, onel=onelg, rand=randg, parf=parfg, actl=actlg
    )


def main_modelgen(config: dict[str, Any], output_path: pathlib.Path) -> None:

    config_sampler = configurator.SamplerConfig(**config["sampler"])
    config_pareto_optimization = configurator.ParetoOptimizationConfig(
        **config["pareto_optimization"]
    )
    quality_estimators_db_path = output_path / QUALITY_ESTIMATOR_DB_FNAME
    cost_estimators_db_path = output_path / COST_ESTIMATOR_DB_FNAME
    quality_estimators_report_path = output_path / QUALITY_ESTIMATOR_REPORT_DIR

    quality_estimators_report_path.mkdir(exist_ok=True, parents=True)

    processing_env = make_bp_config_processing_env(config, output_path)

    old_iter_generator, restart = make_old_iter_generator(
        processing_env.bp_config_db_path
    )
    if restart:
        rng = random.Random(None)
        seed = config_sampler.random_bp_config_rng_seed
        logger.info(f"This is a restart, None is used instead of {seed=}")
    else:
        rng = random.Random(config_sampler.random_bp_config_rng_seed)

    bp_config_unpruned = ptblop.get_unpruned_bp_config(processing_env.model)

    # Two because each block is two possible changes (mlp and attention)
    max_num_changes = round(
        2 * config_sampler.max_num_changes_factor * len(bp_config_unpruned)
    )
    logger.info(
        f"{max_num_changes=} num_blocks={len(bp_config_unpruned)} "
        f"max_num_changes_factor={config_sampler.max_num_changes_factor}"
    )

    processed_bp_config_signatures = set()
    cost_estimator, quality_estimator = None, None
    pareto_front_path, is_new_pareto_front = None, False
    quality_estimator, is_new_quality_estimator = None, False

    bp_config_gens = make_bp_config_generators(
        bp_config_unpruned=bp_config_unpruned,
        min_num_changes=2,
        max_num_changes=max_num_changes,
        processed_bp_config_singatures=processed_bp_config_signatures,
        rng=rng,
    )

    for data_iter in iter_range(config_sampler.n_data_iter):

        iter_generator = make_generator_for_data_iter(
            data_iter,
            config_sampler.n_val_rand,
            config_sampler.trn_data_iter_configs,
            2 * len(bp_config_unpruned),
        )

        for bp_config_type, bp_config_id in iter_generator:
            tmp = next(old_iter_generator)
            bp_config_id_old, bp_config_type_old, bp_config_signature = tmp
            if (
                bp_config_id_old == bp_config_id
                and bp_config_type_old == bp_config_type
            ):
                logger.info(f"SKIPPING {bp_config_id=} {bp_config_type=}")
                processed_bp_config_signatures.add(bp_config_signature)
                if bp_config_type in {"zerl", "onel"}:
                    next(bp_config_gens.get_gen(bp_config_type))
            elif bp_config_id_old is None and bp_config_type_old is None:
                logger.info(f"Processing {bp_config_id=} {bp_config_type=}")
                if bp_config_type == "actl":
                    if quality_estimator is None:
                        quality_estimator = None  # TODO
                    if is_new_quality_estimator:
                        pbcs = processed_bp_config_signatures
                        nc = config_sampler.actl_num_scored_candidates
                        bp_config_gens.actl = make_actl_bp_config_generator(
                            bp_config_unpruned=bp_config_unpruned,
                            min_num_changes=2,
                            max_num_changes=max_num_changes,
                            max_random_config_trials=MAX_RANDOM_CONFIG_TRIALS,
                            processed_bp_bconfig_signatures=pbcs,
                            rng=rng,
                            quality_estimator=quality_estimator,
                            num_scored_candidates=nc,
                        )
                        is_new_quality_estimator = False
                elif bp_config_type == "parf" and is_new_pareto_front:
                    bp_config_gens.parf = make_parf_bp_config_generator(
                        pareto_front_path=pareto_front_path,
                        min_num_changes=2,
                        max_num_changes=max_num_changes,
                        quality_metric=config_sampler.quality_evaluator_metric,
                        min_quality=config_sampler.parf_min_quality_evaluator_metric,
                        processed_bp_bconfig_signatures=processed_bp_config_signatures,
                        rng=rng,
                    )
                    is_new_pareto_front = False
                sample_and_process_bp_config(
                    bp_config_id=bp_config_id,
                    bp_config_type=bp_config_type,
                    data_iter=data_iter,
                    bp_config_gens=bp_config_gens,
                    processing_env=processing_env,
                    processed_bp_config_signatures=processed_bp_config_signatures,
                )
            else:
                msg = f"Schedules differ: {bp_config_id=}, {bp_config_type=} "
                msg += f"vs {bp_config_id_old=}, {bp_config_type_old=}"
                raise ValueError(msg)

        # data_iter == 1 -> validation, we cannot compute predictors + pareto fronts

        if data_iter != 1:
            pf_basename = PARETO_FRONT_FNAME_TEMPLATE % data_iter
            pareto_front_path = output_path / PARETO_FRONT_DIR / pf_basename

            if pareto_front_path.exists():
                msg = f"Pareto front {pareto_front_path} exists, skipping generation"
                logger.info(msg)
            else:
                # Train predictors

                # TODO Put training predictors into in a separate function?
                if cost_estimator is None:
                    cost_estimator_id = "estimator_quality_%04d" % data_iter
                    cost_estimator, cost_estimator_metrics = (
                        estimator_helpers.train_param_estimator(
                            processing_env.bp_config_db_path
                        )
                    )
                    ceid = {"estimator_id": cost_estimator_id}
                    update_db(cost_estimators_db_path, ceid | cost_estimator_metrics)

                quality_estimator_id = "estimator_quality_%04d" % data_iter
                quality_estimator, quality_estimator_metrics = (
                    estimator_helpers.train_quality_estimator(
                        bp_config_db_path=processing_env.bp_config_db_path,
                        quality_metric=config_sampler.quality_evaluator_metric,
                        quality_estimator_id=quality_estimator_id,
                        quality_estimator_report_path=quality_estimators_report_path,
                    )
                )
                is_new_quality_estimator = True
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
                is_new_pareto_front = True
