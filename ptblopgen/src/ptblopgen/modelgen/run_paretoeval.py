import json
import logging
import pathlib
import random
import shutil
from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import ptblop
import torch

from .. import builders, utils
from . import estimator_helpers, pareto_helpers

logger = logging.getLogger(__name__)


@dataclass
class BPConfigProcsessingEvalEnvironment:
    run_id: str
    device: torch.device
    model: torch.nn.Module
    evaluator_fn: Any

    def __init__(self, config: dict[str, Any]):
        self.run_id = utils.make_runid()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model, evaluator_fn = builders.make_model_and_evaluator(
            config["model"], config["evaluator"], self.device
        )
        ptblop.apply_bp_config_in_place(model, {})
        self.model = model
        self.evaluator_fn = evaluator_fn


def make_pareto_evaluated_paths(
    pareto_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    p1 = pareto_path.parent / (pareto_path.stem + "_evaluated.json")
    p2 = pareto_path.parent / (pareto_path.stem + "_evaluated.json.bak")
    return p1, p2


def read_processed_bp_config_signatures(pareto_path: pathlib.Path) -> set[int]:
    processed_bp_config_signatures = set()
    if pareto_path.exists():
        with open(pareto_path, "rt") as f:
            for line in f:
                d = json.loads(line)
                bpcs = utils.bp_config_signature_from_str(d["bp_config_signature"])
                processed_bp_config_signatures.add(bpcs)
    return processed_bp_config_signatures


def read_pareto_front_data(pareto_path: pathlib.Path):
    with open(pareto_path, "rt") as f:
        pf_data_raw = [json.loads(line) for line in f]
    return pf_data_raw


def process_bp_config(
    bp_config,
    processing_env: BPConfigProcsessingEvalEnvironment,
):
    res = {
        "run_id": processing_env.run_id,
    }
    ptblop.apply_bp_config_in_place(
        processing_env.model, bp_config, set_unused_layers_to_none=False
    )
    res["mparams"] = utils.get_num_active_params(processing_env.model) / 1.0e6
    res_evaluation = processing_env.evaluator_fn(
        processing_env.model, processing_env.device
    )
    res |= res_evaluation
    res["timestamp"] = utils.get_timestamp()
    device_str = str(processing_env.device)
    if "cuda" in device_str:
        device_str += " @ " + torch.cuda.get_device_name(processing_env.device)
    res["device"] = device_str
    v_ptblop, v_ptblopgen = utils.get_versions()
    res["ptblop_version"] = v_ptblop
    res["ptblopgen_version"] = v_ptblopgen
    return res


def update_pareto_front(
    db_path: pathlib.Path,
    db_path_bak: pathlib.Path,
    db_entry: dict[str, Any],
) -> None:
    if not db_path.exists():
        with open(db_path, "wt") as f:
            f.write(json.dumps(db_entry) + "\n")
    else:
        # Read data
        with open(db_path, "rt") as f:
            pareto_data = [json.loads(line) for line in f]

        # Append new record
        pareto_data.append(db_entry)

        # Sort
        pareto_data.sort(key=lambda d: -d["mparams_pred"])

        # Copy old data to backup, in case job gets killed or exception occurs
        shutil.copy2(db_path, db_path_bak)

        # Save updated data

        with open(db_path, "wt") as f:
            for di in pareto_data:
                f.write(json.dumps(di) + "\n")


def filter_processed(pareto_front_data, processed_bp_config_signatures):
    res = []
    for d in pareto_front_data:
        bpcs = utils.bp_config_signature_from_str(d["bp_config_signature"])
        if bpcs not in processed_bp_config_signatures:
            res.append(d)
    return res


def filter_pareto_front_by_level(*, pareto_front_data, pareto_level, metric_key):
    bpc_signature_to_level = {}

    cur_data = pareto_front_data

    for cur_pareto_level in range(pareto_level):
        new_data = []
        mparams = np.fromiter((d["mparams_pred"] for d in cur_data), dtype=np.float64)
        metric = np.fromiter((d[metric_key] for d in cur_data), dtype=np.float64)
        mask = pareto_helpers.get_pf_mask(
            o1=mparams, o2=metric, mode=pareto_helpers.Mode.O1_MIN_O2_MAX
        )
        for j in range(len(cur_data)):
            if mask[j]:
                bpc_singature = utils.get_bp_config_signature(cur_data[j]["bp_config"])
                bpc_signature_to_level[bpc_singature] = cur_pareto_level
            else:
                new_data.append(cur_data[j])
        cur_data = new_data

    res = []
    for d in pareto_front_data:
        bpc_signature = utils.get_bp_config_signature(d["bp_config"])
        level = bpc_signature_to_level.get(bpc_signature)
        if level is not None:
            d["pareto_level"] = level
            res.append(d)

    n_tot = len(pareto_front_data)
    logger.info(f"Filtered Pareto front by {pareto_level=} - {n_tot=}")
    return res


def filter_pareto_front(
    *, pareto_front_data, config, min_metric, min_mparams, max_mparams, pareto_level
):
    metric_key = config["sampler"]["quality_evaluator_metric"] + "_pred"
    if pareto_level is not None:
        pareto_front_data = filter_pareto_front_by_level(
            pareto_front_data=pareto_front_data,
            pareto_level=pareto_level,
            metric_key=metric_key,
        )
    if min_metric is not None:

        pareto_front_data = [
            d for d in pareto_front_data if d[metric_key] >= min_metric
        ]
        n_tot = len(pareto_front_data)
        logger.info(f"Filtered Pareto front by min_metric - {n_tot=}")

    if min_mparams is not None:
        pareto_front_data = [
            d for d in pareto_front_data if d["mparams_pred"] >= min_mparams
        ]
        n_tot = len(pareto_front_data)
        logger.info(f"Filtered Pareto front by min_params - {n_tot=}")

    if max_mparams is not None:
        pareto_front_data = [
            d for d in pareto_front_data if d["mparams_pred"] <= max_mparams
        ]
        n_tot = len(pareto_front_data)
        logger.info(f"Filtered Pareto front by max_mparams - {n_tot=}")

    return pareto_front_data


def eval_pareto_front(config, pareto_evaluated_path, pareto_evaluated_plot_path=None):
    pareto_evaluated_path = pathlib.Path(pareto_evaluated_path)
    if pareto_evaluated_plot_path is None:
        plot_name = pareto_evaluated_path.stem + ".png"
        pareto_evaluated_plot_path = pareto_evaluated_path.parent / plot_name

    metric_key = config["sampler"]["quality_evaluator_metric"]
    metric_pred_key = config["sampler"]["quality_evaluator_metric"] + "_pred"
    metric_pred_min_key = config["sampler"]["quality_evaluator_metric"] + "_pred_min"
    metric_pred_max_key = config["sampler"]["quality_evaluator_metric"] + "_pred_max"

    metric_pred, metric_pred_min, metric_pred_max, metric_true = [], [], [], []
    mparams = []

    with open(pareto_evaluated_path, "rt") as f:
        for line in f:
            d = json.loads(line)
            metric_pred.append(d[metric_pred_key])
            metric_pred_min.append(d[metric_pred_min_key])
            metric_pred_max.append(d[metric_pred_max_key])
            metric_true.append(d["evaluation"][metric_key])
            mparams.append(d["evaluation"]["mparams"])

    if len(metric_pred) <= 3:
        logger.info("Fewer than 3 points in Pareto front, skipping evaluation")
        return

    metric_pred = np.array(metric_pred)
    metric_pred_min = np.array(metric_pred_min)
    metric_pred_max = np.array(metric_pred_max)
    metric_true = np.array(metric_true)
    err_pred = 0.5 * np.abs(metric_pred_max - metric_pred_min)
    err_true = np.abs(metric_true - metric_pred)

    stats = estimator_helpers.get_metrics(
        y=metric_true, ypred=metric_pred, err=err_true, errpred=err_pred, prefix=""
    )

    ALPHA = 1.0

    CTRUE = "#2ca02c"
    CPRED = "#1f77b4"
    CPRED_CONF = "#aec7e8"
    CERR_CMP = "#ff7f0e"

    fig, axs = plt.subplots(figsize=(18, 12), nrows=2, ncols=2)
    axs[0, 0].set_title(f"mparams vs {metric_key}")
    axs[0, 0].set_ylabel(f"{metric_key}")
    axs[0, 0].set_xlabel("mparams")
    axs[0, 0].fill_between(mparams, metric_pred_min, metric_pred_max, color=CPRED_CONF)
    axs[0, 0].scatter(mparams, metric_pred, alpha=ALPHA, c=CPRED)
    axs[0, 0].scatter(mparams, metric_true, alpha=ALPHA, c=CTRUE)
    axs[0, 0].grid()

    axs[1, 0].set_title(f"{metric_key}_true vs {metric_pred_key}")
    axs[1, 0].axline((metric_true[-1], metric_true[-1]), slope=1, c="black")
    axs[1, 0].set_xlabel(f"{metric_key}_true")
    axs[1, 0].set_ylabel(f"{metric_pred_key}")
    # ii = np.argsort(metric_true)
    # axs[1, 0].fill_between(
    #     metric_true[ii], metric_pred_min[ii], metric_pred_max[ii], color=CPRED_CONF
    # )
    err1 = np.maximum(metric_pred - metric_pred_min, 0.0)
    err2 = np.maximum(metric_pred_max - metric_pred, 0.0)
    tmp_err = np.vstack((err1, err2))
    axs[1, 0].errorbar(
        x=metric_true,
        y=metric_pred,
        yerr=tmp_err,
        fmt="none",
        color=CPRED,
        elinewidth=4,
        zorder=1,
    )
    axs[1, 0].scatter(metric_true, metric_pred, alpha=ALPHA, c=CPRED)
    axs[1, 0].grid()

    axs[1, 1].set_title(f"{metric_key}_err_true vs {metric_key}_err_pred")
    axs[1, 1].set_xlabel(f"{metric_key}_err_true")
    axs[1, 1].set_ylabel(f"{metric_key}_err_pred")
    axs[1, 1].scatter(err_true, err_pred, alpha=ALPHA, c=CERR_CMP)
    axs[1, 1].grid()

    num = stats["n"]
    rms = stats["rms"]
    mae = stats["mae"]
    cor = stats["cor"]
    spr = stats["spr"]
    corerr = stats["corerr"]
    sprerr = stats["sprerr"]
    stats_str = f"n = {num}\n\nrms = {rms:.3f}\nmae = {mae:.3f}\n\n"
    stats_str += f"cor = {cor:.3f}\nspr = {spr:.3f}\n\n"
    stats_str += f"corerr = {corerr:.3f}\nsprerr = {sprerr:.3f}"
    axs[0, 1].text(0.2, 0.2, stats_str, fontsize=18)
    axs[0, 1].set_xlim(0.0, 1.0)
    axs[0, 1].set_ylim(0.0, 1.0)
    axs[0, 1].set_axis_off()

    logger.info(f"Pareto front plots saved to {pareto_evaluated_plot_path}")
    if pareto_evaluated_plot_path is not None:
        fig.savefig(pareto_evaluated_plot_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main_paretoeval(
    *,
    config: dict[str, Any],
    pareto_path: pathlib.Path,
    min_metric: Optional[float],
    min_mparams: Optional[float],
    max_mparams: Optional[float],
    shuffle: bool,
    pareto_level: int,
) -> None:

    pareto_front_data = read_pareto_front_data(pareto_path)
    n_tot = len(pareto_front_data)
    logger.info(f"Read {n_tot=} configurations from {pareto_path}")

    pareto_evaluated_path, pareto_evaluated_path_bak = make_pareto_evaluated_paths(
        pareto_path
    )
    processed_bp_config_signatures = read_processed_bp_config_signatures(
        pareto_evaluated_path
    )
    n_evaluated = len(processed_bp_config_signatures)
    logger.info(f"Read {n_evaluated=} from {pareto_evaluated_path}")

    processing_env = BPConfigProcsessingEvalEnvironment(config)
    pareto_front_data = filter_processed(
        pareto_front_data, processed_bp_config_signatures
    )
    n_tot = len(pareto_front_data)
    logger.info(f"Filtered processed Pareto front data - {n_tot=}")

    pareto_front_data = filter_pareto_front(
        pareto_front_data=pareto_front_data,
        config=config,
        min_metric=min_metric,
        min_mparams=min_mparams,
        max_mparams=max_mparams,
        pareto_level=pareto_level,
    )
    n_tot = len(pareto_front_data)

    if shuffle:
        logger.info("Shuffling pareto front data before processing")
        random.shuffle(pareto_front_data)
    else:
        logger.info("Skipping shuffling pareto front data")

    for i, d in enumerate(pareto_front_data, start=1):
        bpcs = utils.bp_config_signature_from_str(d["bp_config_signature"])
        if bpcs in processed_bp_config_signatures:
            logger.info(f"Skipping bp_config {bpcs}")
        else:
            logger.info(f"Processing bp_config {i} out of {n_tot} - {bpcs=}")
            bp_config = d["bp_config"]
            res = process_bp_config(bp_config, processing_env)
            d["evaluation"] = res
            update_pareto_front(pareto_evaluated_path, pareto_evaluated_path_bak, d)
            signature = utils.get_bp_config_signature(bp_config)
            processed_bp_config_signatures.add(signature)
        eval_pareto_front(config, pareto_evaluated_path)
    pareto_evaluated_path_bak.unlink(missing_ok=True)
