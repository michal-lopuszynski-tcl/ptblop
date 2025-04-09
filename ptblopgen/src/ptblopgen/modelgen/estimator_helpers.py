import json
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

from .. import estimators, utils

logger = logging.getLogger(__name__)


# Universal functions


def read_data_one_file(db_path, data_iter):

    def __is_val(d):
        return d["id"].startswith("val.")

    data_raw = []

    with open(db_path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if d["status"] == "ok":
                data_raw.append(d)
    if data_iter > 0:
        data = [d for d in data_raw if d["data_iter"] <= data_iter]
    else:
        data = data_raw

    if len(data) < len(data_raw):
        max_data_iter = max(d["data_iter"] for d in data)
        n_raw, n = len(data_raw), len(data)
        msg = f"Estimator data loader: using {n=} instead of {n_raw=}"
        msg += f" available configs - {max_data_iter=} vs. used {data_iter=} "
        logger.warning(msg)

    data_val = [d for d in data if __is_val(d)]
    data_trn = [d for d in data if not __is_val(d)]
    return data_trn, data_val


def read_data(db_paths, data_iter):
    data_trn, data_val = [], []
    duplicates_trn, duplicates_val = 0, 0

    processed_bp_config_signatures = set()

    for cur_db_path in db_paths:
        cur_data_trn, cur_data_val = read_data_one_file(cur_db_path, data_iter)
        cur_duplicates_trn, cur_duplicates_val = 0, 0

        for d in cur_data_val:
            bp_config = d["bp_config"]
            bp_signature = utils.get_bp_config_signature(bp_config)
            if bp_signature not in processed_bp_config_signatures:
                data_val.append(d)
                processed_bp_config_signatures.add(bp_signature)
            else:
                cur_duplicates_val += 1

        for d in cur_data_trn:
            bp_config = d["bp_config"]
            bp_signature = utils.get_bp_config_signature(bp_config)
            if bp_signature not in processed_bp_config_signatures:
                data_trn.append(d)
                processed_bp_config_signatures.add(bp_signature)
            else:
                cur_duplicates_trn += 1

        duplicates_trn += cur_duplicates_trn
        duplicates_val += cur_duplicates_val
        cur_n_trn = len(cur_data_trn)
        cur_n_val = len(cur_data_val)
        logger.info(
            f"Read {cur_db_path}, {cur_n_trn=}, {cur_n_val}, "
            f"{cur_duplicates_trn=}, {cur_duplicates_val=}"
        )
    logger.info(
        f"Final trn datasize = {len(data_trn)}, "
        f"after removing {duplicates_trn} duplicates"
    )
    logger.info(
        f"Final val datasize = {len(data_val)}, "
        f"after removing {duplicates_val} duplicates"
    )
    return data_trn, data_val


def get_target(data, target_column):
    res = []
    for d in data:
        tmp = d
        for k in target_column.split("."):
            tmp = tmp[k]
        res.append(tmp)
    return np.array(res)


# Evaluation - bounds regressor


def _get_bounds_pred(bounds_regressor, X, y):
    ypred, ypred_min, ypred_max = bounds_regressor.predict_with_bounds(X)
    errpred = 0.5 * np.abs(ypred_max - ypred_min)
    err = np.abs(ypred - y)
    return ypred, ypred_min, ypred_max, err, errpred


def get_metrics(*, prefix, y, ypred, err, errpred):
    rms = sklearn.metrics.root_mean_squared_error(y, ypred)
    mae = sklearn.metrics.mean_absolute_error(y, ypred)
    corerr = np.corrcoef(err, errpred)[0][1]
    return {
        f"{prefix}rms": float(rms),
        f"{prefix}mae": float(mae),
        f"{prefix}corerr": float(corerr),
        f"{prefix}n": len(y),
    }


def evaluate_bounds_estimator(
    *, bounds_regressor, X_trn, y_trn, X_val, y_val, plot_fname=None
):

    y_trn_pred, y_trn_pred_min, y_trn_pred_max, err_trn, err_trn_pred = (
        _get_bounds_pred(bounds_regressor, X_trn, y_trn)
    )

    r_trn = get_metrics(
        prefix="trn_", y=y_trn, ypred=y_trn_pred, err=err_trn, errpred=err_trn_pred
    )

    y_val_pred, y_val_pred_min, y_val_pred_max, err_val, err_val_pred = (
        _get_bounds_pred(bounds_regressor, X_val, y_val)
    )

    r_val = get_metrics(
        prefix="val_", y=y_val, ypred=y_val_pred, err=err_val, errpred=err_val_pred
    )
    if plot_fname is not None:
        fig, axs = plt.subplots(
            figsize=(18, 12), nrows=2, ncols=3, sharex="col", sharey="col"
        )
        ALPHA = 0.2
        CTRN = "#1f77b4"
        CVAL = "#2ca02c"

        axs[0, 0].set_title("y_trn_true vs y_trn_pred")
        axs[0, 0].axline((y_trn[0], y_trn[0]), slope=1, c="black")
        axs[0, 0].set_ylabel("y_pred")

        # # Errorbars
        # x = np.vstack((y_trn, y_trn))
        # y = np.vstack((y_trn_pred_min, y_trn_pred_max))
        # axs[0, 0].plot(x, y, "-", c=CTRN, alpha=ALPHA)

        axs[0, 0].scatter(y_trn, y_trn_pred, alpha=ALPHA, c=CTRN)
        axs[0, 0].grid()

        axs[0, 1].set_title("err_trn_true vs err_trn_pred")
        axs[0, 1].scatter(err_trn, err_trn_pred, alpha=ALPHA, c=CTRN)
        axs[0, 1].axline((np.min(err_trn), np.min(err_trn)), slope=1, c="black")
        axs[0, 1].grid()
        axs[0, 1].set_ylabel("err_pred")

        axs[1, 0].set_title("y_val_true vs y_val_pred")
        axs[1, 0].axline((y_val[0], y_val[0]), slope=1, c="black")
        axs[1, 0].scatter(y_val, y_val_pred, alpha=ALPHA, c=CVAL)
        axs[1, 0].set_xlabel("y_true")
        axs[1, 0].set_ylabel("y_pred")

        # # Errorbars
        # x = np.vstack((y_val, y_val))
        # y = np.vstack((y_val_pred_min, y_val_pred_max))
        # axs[1, 0].plot(x, y, "-", c=CVAL, alpha=ALPHA)

        axs[1, 0].grid()

        axs[1, 1].set_title("err_val_true vs err_val_pred")
        axs[1, 1].scatter(err_val, err_val_pred, alpha=ALPHA, c=CVAL)
        axs[1, 1].axline((np.min(err_val), np.min(err_val)), slope=1, c="black")
        axs[1, 1].grid()
        axs[1, 1].set_xlabel("err_true")
        axs[1, 1].set_ylabel("err_pred")
        axs[0, 2].set_axis_off()
        num = r_trn["trn_n"]
        rms = r_trn["trn_rms"]
        mae = r_trn["trn_mae"]
        corerr = r_trn["trn_corerr"]
        trn_stats = f"trn_n = {num}\ntrn_rms = {rms:.3f}\n"
        trn_stats += f"trn_mae = {mae:.3f}\ntrn_corerr = {corerr:.3f}"

        axs[0, 2].text(0.2, 0.7, trn_stats, fontsize=18)
        num = r_val["val_n"]
        rms = r_val["val_rms"]
        mae = r_val["val_mae"]
        corerr = r_val["val_corerr"]
        val_stats = f"val_n = {num}\nval_rms = {rms:.3f}\n"
        val_stats += f"val_mae = {mae:.3f}\nval_corerr = {corerr:.3f}"

        axs[1, 2].text(0.2, 0.7, val_stats, fontsize=18)
        axs[1, 2].set_axis_off()
        fig.savefig(plot_fname, dpi=240, bbox_inches="tight")
        plt.close(fig)

    # r_trn = _evaluate_bounds_regressor_single_dataset(
    #     bounds_regressor=bounds_regressor, X=X_trn, y=y_trn, prefix="trn_"
    # )
    # r_val = _evaluate_bounds_regressor_single_dataset(
    #     bounds_regressor=bounds_regressor, X=X_val, y=y_val, prefix="val_"
    # )
    return r_trn | r_val


# Evaluation - simple regressor


def _evaluate_regressor_single_dataset(*, bounds_regressor, X, y, prefix):
    ypred = bounds_regressor.predict(X)

    rms = sklearn.metrics.root_mean_squared_error(y, ypred)
    mae = sklearn.metrics.mean_absolute_error(y, ypred)
    return {
        f"{prefix}rms": float(rms),
        f"{prefix}mae": float(mae),
    }


def evaluate_estimator(*, bounds_regressor, X_trn, y_trn, X_val, y_val):
    r_trn = _evaluate_regressor_single_dataset(
        bounds_regressor=bounds_regressor, X=X_trn, y=y_trn, prefix="trn_"
    )
    r_val = _evaluate_regressor_single_dataset(
        bounds_regressor=bounds_regressor, X=X_val, y=y_val, prefix="val_"
    )
    return r_trn | r_val


# Quality regressor


def get_quality_feature_names(bp_configs):
    feature_names = []

    for entry in bp_configs[0].keys():
        feature_names.append(f"{entry}_use_attention")
        feature_names.append(f"{entry}_use_mlp")
    return feature_names


def get_quality_features(bp_configs, full_block_mode):

    def __cast(v):
        return float(v)

    feature_list = []
    if not full_block_mode:
        for bpc in bp_configs:
            row = []
            # bp_configs[0] just in case other configs have different keys order
            for entry in bp_configs[0]:
                f1 = __cast(bpc[entry]["use_attention"])
                row.append(f1)
                f2 = __cast(bpc[entry]["use_mlp"])
                row.append(f2)
            feature_list.append(row)
    else:
        for bpc in bp_configs:
            row = []
            # bp_configs[0] just in case other configs have different keys order
            for entry in bp_configs[0]:
                assert bpc[entry]["use_attention"] == bpc[entry]["use_mlp"]
                row.append(__cast(bpc[entry]["use_attention"]))
            feature_list.append(row)
    features = np.array(feature_list)
    return features


# Utils for params "regression"


def _get_params_features_from_bp_config(bp_config):
    n_full, n_attention, n_mlp = 0.0, 0.0, 0.0
    for d in bp_config.values():
        use_attention, use_mlp = d["use_attention"], d["use_mlp"]
        if use_attention and use_mlp:
            n_full += 1.0
        elif use_mlp:
            n_mlp += 1.0
        elif use_attention:
            n_attention += 1.0
    return [n_full, n_attention, n_mlp]


def get_params_features(bp_configs):
    return np.array([_get_params_features_from_bp_config(bpc) for bpc in bp_configs])


# Main library functionality


def train_quality_estimator(
    *,
    bp_config_db_paths: pathlib.Path,
    quality_estimator_report_path: pathlib.Path,
    quality_estimators_db_path: pathlib.Path,
    data_iter: int,
    quality_metric: str,
    run_id: str,
    full_block_mode: bool,
):
    estimator_kwargs = None  # TODO: make it a parameter

    suffix = f"{utils.get_timestamp_for_fname()}_{utils.get_random_str(6)}"
    estimator_id = f"qual{data_iter:04d}_{suffix}"

    data_trn, data_val = read_data(bp_config_db_paths, data_iter)
    X_trn = get_quality_features([d["bp_config"] for d in data_trn], full_block_mode)
    y_trn = get_target(data_trn, "evaluation." + quality_metric)
    logger.info(f"{X_trn.shape=} {X_trn.dtype=} {y_trn.shape=} {y_trn.dtype=}")

    X_val = get_quality_features([d["bp_config"] for d in data_val], full_block_mode)
    y_val = get_target(data_val, "evaluation." + quality_metric)
    logger.info(f"{X_val.shape=} {X_val.dtype=} {y_val.shape=} {y_val.dtype=}")

    n_examples_trn, n_features_trn = X_trn.shape
    n_examples_val, n_features_val = X_val.shape

    # GBM Estimator
    # reg_type = "QuantileGradientBoostingBoundsEstimator"
    # reg_kwargs = dict(
    #     learning_rate=0.03,
    #     n_estimators=200,
    #     max_depth=6,
    #     min_samples_leaf=9,
    #     min_samples_split=9,
    # )
    # reg = estimators.QuantileGradientBoostingBoundsEstimator(**reg_kwargs)
    # reg.fit(X_trn, y_trn)

    # Linear Estimator
    # reg_type = "QuantileLinearEstimator"
    # reg_kwargs = dict(fit_intercept=True, alpha=0.01)
    # reg = blockprunekit.regressors.QuantileLinearEstimator(**reg_kwargs)

    # # Random forest regressor - the default
    # estimator_type = "EnsembleRandomForestBoundsEstimator"
    # if estimator_kwargs is None:
    #     estimator_kwargs = dict(n_regressors=20, n_estimators=300)
    # # if reg_kwargs is None:
    # #     reg_kwargs = dict(
    # #         n_regressors=20, n_estimators=300, max_features=0.8, min_samples_leaf=2
    # #     )
    # estimator = estimators.EnsembleRandomForestBoundsEstimator(**estimator_kwargs)
    # estimator.fit(X_trn, y_trn)

    # Random forest regressor
    estimator_type = "QuantileTabPFNEstimator"
    if estimator_kwargs is None:
        estimator_kwargs = dict(n_estimators=8, q_min=0.2, q_max=0.8)
    # if reg_kwargs is None:
    #     reg_kwargs = dict(
    #         n_regressors=20, n_estimators=300, max_features=0.8, min_samples_leaf=2
    #     )
    estimator = estimators.QuantileTabPFNEstimator(**estimator_kwargs)
    estimator.fit(X_trn, y_trn)

    # Extra trees regressor
    # reg_type = "EnsembleExtraTreesBoundsEstimator"
    # if reg_kwargs is None:
    #     reg_kwargs = dict(
    #         n_regressors=20, n_estimators=300, max_features=0.8, bootstrap=True
    #     )
    # reg = blockprunekit.regressors.EnsembleExtraTreesBoundsEstimator(**reg_kwargs)
    # reg.fit(X_trn, y_trn)

    # # Experimental Ensemble bradeint boosting regressor
    # reg_type = "EnsembleGradientBoostingBoundsEstimator"
    # reg_kwargs = dict(
    #     n_regressors=20,
    #     subsample=0.1,
    #     learning_rate=0.03,
    #     n_estimators=200,
    #     max_depth=6,
    #     min_samples_leaf=9,
    #     min_samples_split=9,
    # )
    # reg = blockprunekit.regressors.EnsembleGradientBoostingBoundsEstimator(
    #     **reg_kwargs
    # )
    # reg.fit(X_trn, y_trn)

    # Saving Estimator - not all regressors support that yet
    # regressor_path = regressor_dir / "quality_regressor.json.gz"
    # blockprunekit.regressors.save_regressor(regressor_path, reg)

    plot_fname = quality_estimator_report_path / f"{estimator_id}.png"
    estimator_metrics = evaluate_bounds_estimator(
        bounds_regressor=estimator,
        X_trn=X_trn,
        y_trn=y_trn,
        X_val=X_val,
        y_val=y_val,
        plot_fname=plot_fname,
    )
    logger.info(f"{estimator_metrics=}")

    # reg1 = blockprunekit.regressors.load_regressor("quality_regressor.json.gz")
    # reg_metrics1 = estimator_helpers.evaluate_bounds_regressor(
    #     bounds_regressor=reg1,
    #     X_trn=X_trn,
    #     y_trn=y_trn,
    #     X_val=X_val,
    #     y_val=y_val,
    #     plot_fname="tmp.png",
    # )
    # logger.info(f"{reg_metrics1=}")
    v_ptblop, v_ptblopgen = utils.get_versions()

    estimator_data = {
        "run_id": run_id,
        "estimator_id": estimator_id,
        "n_examples_trn": n_examples_trn,
        "n_features_trn": n_features_trn,
        "n_examples_val": n_examples_val,
        "n_features_val": n_features_val,
        "estimator_type": estimator_type,
        "estimator_kwargs": estimator_kwargs,
        "estimator_metrics": estimator_metrics,
        "timestamp": utils.get_timestamp(),
        "ptblop_version": v_ptblop,
        "ptblopgen_version": v_ptblopgen,
    }
    logger.info(f"{estimator_data=}")
    utils.update_db(quality_estimators_db_path, estimator_data)
    return estimator, estimator_data, estimator_id


def train_param_estimator(
    *,
    bp_config_db_paths: list[pathlib.Path],
    cost_estimators_db_path: pathlib.Path,
    data_iter: int,
    run_id: str,
    full_block_mode: bool,
):
    suffix = f"{utils.get_timestamp_for_fname()}_{utils.get_random_str(6)}"
    estimator_id = f"cost{data_iter:04d}_{suffix}"

    data_trn, data_val = read_data(bp_config_db_paths, data_iter)
    X_trn = get_quality_features([d["bp_config"] for d in data_trn], full_block_mode)
    y_trn = get_target(data_trn, "mparams")
    logger.info(f"{X_trn.shape=} {X_trn.dtype=} {y_trn.shape=} {y_trn.dtype=}")
    X_val = get_quality_features([d["bp_config"] for d in data_val], full_block_mode)
    y_val = get_target(data_val, "mparams")
    logger.info(f"{X_val.shape=} {X_val.dtype=} {y_val.shape=} {y_val.dtype=}")

    n_examples_trn, n_features_trn = X_trn.shape
    n_examples_val, n_features_val = X_val.shape

    reg_type = "LinearRegression"
    reg_kwargs = dict(fit_intercept=True)
    estimator = sklearn.linear_model.LinearRegression(**reg_kwargs)
    estimator.fit(X_trn, y_trn)

    estimator_metrics = evaluate_estimator(
        bounds_regressor=estimator,
        X_trn=X_trn,
        y_trn=y_trn,
        X_val=X_val,
        y_val=y_val,
    )
    v_ptblop, v_ptblopgen = utils.get_versions()

    estimator_data = {
        "run_id": run_id,
        "estimator_id": estimator_id,
        "n_examples_trn": n_examples_trn,
        "n_features_trn": n_features_trn,
        "n_examples_val": n_examples_val,
        "n_features_val": n_features_val,
        "estimator_type": reg_type,
        "estimator_kwargs": reg_kwargs,
        "estimator_metrics": estimator_metrics,
        "timestamp": utils.get_timestamp(),
        "ptblop_version": v_ptblop,
        "ptblopgen_version": v_ptblopgen,
    }
    logger.info(f"{estimator_data=}")
    utils.update_db(cost_estimators_db_path, estimator_data)

    return estimator, estimator_data, estimator_id
