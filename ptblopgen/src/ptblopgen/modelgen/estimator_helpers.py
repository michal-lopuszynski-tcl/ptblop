import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import ptblop
import sklearn.metrics

from .. import _version, estimators

logger = logging.getLogger(__name__)


# Universal functions


def read_data(db_path):

    def __is_val(d):
        return d["id"].startswith("val.")

    # signatures = set()
    data = []
    with open(db_path, "rt") as f:
        for line in f:
            d = json.loads(line)
            # bpconfig = d["bpconfig"]
            # sig = get_bpconfig_signature(bpconfig)
            # signatures.add(sig)
            data.append(d)

    # logger.info(f"Read {len(signatures)} configurations and stored their signatures")

    data_val = [d for d in data if __is_val(d)]
    data_trn = [d for d in data if not __is_val(d)]
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


def _evaluate_bounds_regressor_single_dataset2(*, bounds_regressor, X, y, prefix):
    ypred, ypred_min, ypred_max = bounds_regressor.predict_with_bounds(X)
    errpred = np.abs(ypred_max - ypred_min)
    err = np.abs(ypred - y)

    rms = sklearn.metrics.root_mean_squared_error(y, ypred)
    mae = sklearn.metrics.mean_absolute_error(y, ypred)
    corerr = np.corrcoef(err, errpred)[0][1]
    return {
        f"{prefix}rms": float(rms),
        f"{prefix}mae": float(mae),
        f"{prefix}corerr": float(corerr),
    }


def evaluate_bounds_regressor2(*, bounds_regressor, X_trn, y_trn, X_val, y_val):
    r_trn = _evaluate_bounds_regressor_single_dataset2(
        bounds_regressor=bounds_regressor, X=X_trn, y=y_trn, prefix="trn_"
    )
    r_val = _evaluate_bounds_regressor_single_dataset2(
        bounds_regressor=bounds_regressor, X=X_val, y=y_val, prefix="val_"
    )
    return r_trn | r_val


def _get_bounds_pred(bounds_regressor, X, y):
    ypred, ypred_min, ypred_max = bounds_regressor.predict_with_bounds(X)
    errpred = np.abs(ypred_max - ypred_min)
    err = np.abs(ypred - y)
    return ypred, ypred_min, ypred_max, err, errpred


def _get_metrics(*, prefix, y, ypred, err, errpred):
    rms = sklearn.metrics.root_mean_squared_error(y, ypred)
    mae = sklearn.metrics.mean_absolute_error(y, ypred)
    corerr = np.corrcoef(err, errpred)[0][1]
    return {
        f"{prefix}rms": float(rms),
        f"{prefix}mae": float(mae),
        f"{prefix}corerr": float(corerr),
    }


def evaluate_bounds_estimator(
    *, bounds_regressor, X_trn, y_trn, X_val, y_val, plot_fname=None
):

    y_trn_pred, y_trn_pred_min, y_trn_pred_max, err_trn, err_trn_pred = (
        _get_bounds_pred(bounds_regressor, X_trn, y_trn)
    )

    r_trn = _get_metrics(
        prefix="trn_", y=y_trn, ypred=y_trn_pred, err=err_trn, errpred=err_trn_pred
    )

    y_val_pred, y_val_pred_min, y_val_pred_max, err_val, err_val_pred = (
        _get_bounds_pred(bounds_regressor, X_val, y_val)
    )

    r_val = _get_metrics(
        prefix="val_", y=y_val, ypred=y_val_pred, err=err_val, errpred=err_val_pred
    )
    if plot_fname is not None:
        fig, axs = plt.subplots(
            figsize=(18, 12), nrows=2, ncols=3, sharex="col", sharey="col"
        )
        ALPHA = 0.2
        CTRN = "#1f77b4"
        CVAL = "#2ca02c"

        axs[0, 0].set_title("y_trn vs y_trn_pred")
        axs[0, 0].axline((y_trn[0], y_trn[0]), slope=1, c="black")

        # # Errorbars
        # x = np.vstack((y_trn, y_trn))
        # y = np.vstack((y_trn_pred_min, y_trn_pred_max))
        # axs[0, 0].plot(x, y, "-", c=CTRN, alpha=ALPHA)

        axs[0, 0].scatter(y_trn, y_trn_pred, alpha=ALPHA, c=CTRN)
        axs[0, 0].grid()

        axs[0, 1].set_title("err_trn vs err_trn_pred")
        axs[0, 1].scatter(err_trn, err_trn_pred, alpha=ALPHA, c=CTRN)
        axs[0, 1].grid()

        axs[1, 0].set_title("y_val vs y_val_pred")
        axs[1, 0].axline((y_val[0], y_val[0]), slope=1, c="black")
        axs[1, 0].scatter(y_val, y_val_pred, alpha=ALPHA, c=CVAL)

        # # Errorbars
        # x = np.vstack((y_val, y_val))
        # y = np.vstack((y_val_pred_min, y_val_pred_max))
        # axs[1, 0].plot(x, y, "-", c=CVAL, alpha=ALPHA)

        axs[1, 0].grid()

        axs[1, 1].set_title("err_val vs err_val_pred")
        axs[1, 1].scatter(err_val, err_val_pred, alpha=ALPHA, c=CVAL)
        axs[1, 1].grid()
        axs[0, 2].set_axis_off()
        rms = r_trn["trn_rms"]
        mae = r_trn["trn_mae"]
        corerr = r_trn["trn_corerr"]
        trn_stats = (
            f"trn_rms = {rms:.3f}\ntrn_mae = {mae:.3f}\ntrn_corerr = {corerr:.3f}"
        )
        axs[0, 2].text(0.2, 0.8, trn_stats, fontsize=18)
        rms = r_val["val_rms"]
        mae = r_val["val_mae"]
        corerr = r_val["val_corerr"]
        val_stats = (
            f"val_rms = {rms:.3f}\nval_mae = {mae:.3f}\nval_corerr = {corerr:.3f}"
        )
        axs[1, 2].text(0.2, 0.8, val_stats, fontsize=18)
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


def get_quality_features(bpconfigs):

    def __cast(v):
        return float(v)

    feature_list = []

    for bpc in bpconfigs:
        row = []
        # bpconfigs[0] just in case other configs have different keys order
        for entry in bpconfigs[0]:
            f1 = __cast(bpc[entry]["use_attention"])
            row.append(f1)
            f2 = __cast(bpc[entry]["use_mlp"])
            row.append(f2)
        feature_list.append(row)

    features = np.array(feature_list)
    return features


# Utils for params "regression"


def _get_params_features_from_bpconfig(bpconfig):
    n_full, n_attention, n_mlp = 0.0, 0.0, 0.0
    for d in bpconfig.values():
        use_attention, use_mlp = d["use_attention"], d["use_mlp"]
        if use_attention and use_mlp:
            n_full += 1.0
        elif use_mlp:
            n_mlp += 1.0
        elif use_attention:
            n_attention += 1.0
    return [n_full, n_attention, n_mlp]


def get_params_features(bpconfigs):
    return np.array([_get_params_features_from_bpconfig(bpc) for bpc in bpconfigs])


# Main library functionality


def train_quality_estimator(
    bp_config_db_path,
    quality_metric,
    quality_estimator_id,
    quality_estimator_report_path,
    reg_kwargs=None,
):
    data_trn, data_val = read_data(bp_config_db_path)
    X_trn = get_quality_features([d["bp_config"] for d in data_trn])
    y_trn = get_target(data_trn, "evaluation." + quality_metric)
    logger.info(f"{X_trn.shape=} {X_trn.dtype=} {y_trn.shape=} {y_trn.dtype=}")

    X_val = get_quality_features([d["bp_config"] for d in data_val])
    y_val = get_target(data_val, "evaluation." + quality_metric)
    logger.info(f"{X_val.shape=} {X_val.dtype=} {y_val.shape=} {y_val.dtype=}")

    n_examples_trn, n_features_trn = X_trn.shape
    n_examples_val, n_features_val = X_val.shape

    # GBM Estimator
    reg_type = "QuantileGradientBoostingBoundsEstimator"
    reg_kwargs = dict(
        learning_rate=0.03,
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=9,
        min_samples_split=9,
    )
    reg = estimators.QuantileGradientBoostingBoundsEstimator(**reg_kwargs)
    reg.fit(X_trn, y_trn)

    # Linear Estimator
    # reg_type = "QuantileLinearEstimator"
    # reg_kwargs = dict(fit_intercept=True, alpha=0.01)
    # reg = blockprunekit.regressors.QuantileLinearEstimator(**reg_kwargs)

    # Random forest regressor
    # reg_type = "EnsembleRandomForestBoundsEstimator"
    # if reg_kwargs is None:
    #     reg_kwargs = dict(n_regressors=20, n_estimators=300)
    # # if reg_kwargs is None:
    # #     reg_kwargs = dict(
    # #         n_regressors=20, n_estimators=300, max_features=0.8, min_samples_leaf=2
    # #     )
    # reg = blockprunekit.regressors.EnsembleRandomForestBoundsEstimator(**reg_kwargs)
    # reg.fit(X_trn, y_trn)

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

    plot_fname = quality_estimator_report_path / f"{quality_estimator_id}.png"
    reg_metrics = evaluate_bounds_estimator(
        bounds_regressor=reg,
        X_trn=X_trn,
        y_trn=y_trn,
        X_val=X_val,
        y_val=y_val,
        plot_fname=plot_fname,
    )
    logger.info(f"{reg_metrics=}")

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

    reg_data = {
        "n_examples_trn": n_examples_trn,
        "n_features_trn": n_features_trn,
        "n_examples_val": n_examples_val,
        "n_features_val": n_features_val,
        "regressor_type": reg_type,
        "regressor_kwargs": reg_kwargs,
        "regressor_metrics": reg_metrics,
        "ptblop_version": ptblop.__version__,
        "ptblopgen_version": _version.__version__,
    }
    logger.info(f"{reg_data=}")
    return reg, reg_data


def train_param_estimator(bp_config_db_path):
    data_trn, data_val = read_data(bp_config_db_path)
    X_trn = get_quality_features([d["bp_config"] for d in data_trn])
    y_trn = get_target(data_trn, "mparams")
    logger.info(f"{X_trn.shape=} {X_trn.dtype=} {y_trn.shape=} {y_trn.dtype=}")
    X_val = get_quality_features([d["bp_config"] for d in data_val])
    y_val = get_target(data_val, "mparams")
    logger.info(f"{X_val.shape=} {X_val.dtype=} {y_val.shape=} {y_val.dtype=}")

    n_examples_trn, n_features_trn = X_trn.shape
    n_examples_val, n_features_val = X_val.shape

    reg_type = "LinearRegression"
    reg_kwargs = dict(fit_intercept=True)
    reg = sklearn.linear_model.LinearRegression(**reg_kwargs)
    reg.fit(X_trn, y_trn)

    reg_metrics = evaluate_estimator(
        bounds_regressor=reg,
        X_trn=X_trn,
        y_trn=y_trn,
        X_val=X_val,
        y_val=y_val,
    )

    reg_data = {
        "n_examples_trn": n_examples_trn,
        "n_features_trn": n_features_trn,
        "n_examples_val": n_examples_val,
        "n_features_val": n_features_val,
        "regressor_type": reg_type,
        "regressor_kwargs": reg_kwargs,
        "regressor_metrics": reg_metrics,
        "ptblop_version": ptblop.__version__,
        "ptblopgen_version": _version.__version__,
    }
    logger.info(f"{reg_data=}")
    return reg, reg_data
