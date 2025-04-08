import abc
import base64
import gzip
import json
import pathlib
from typing import IO, Any

import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import skops.io
import tabpfn

from .. import utils


def _get_name_from_object(o: Any) -> str:
    return type(o).__module__ + "." + type(o).__name__


def _get_name_from_type(t: type) -> str:
    return t.__module__ + "." + t.__name__


def _open(p: pathlib.Path, mode: str) -> IO[Any] | gzip.GzipFile:
    if str(p).endswith(".gz"):
        return gzip.open(p, mode)
    else:
        return open(p, mode)


def _sklearn_to_str(o: Any) -> str:
    o_bytes = skops.io.dumps(o)
    o_ascii = base64.b64encode(o_bytes).decode("ascii")
    return o_ascii


def _str_to_sklearn(o_ascii: str) -> Any:
    trusted_types = [
        "_loss.CyPinballLoss",
        "sklearn._loss.link.IdentityLink",
        "sklearn._loss.link.Interval",
        "sklearn._loss.loss.PinballLoss",
    ]

    o_bytes = base64.b64decode(o_ascii.encode("ascii"))
    o = skops.io.loads(o_bytes, trusted=trusted_types)
    return o


class BoundsEstimator(abc.ABC):

    @abc.abstractmethod
    def fit(self, X, y) -> None:
        pass

    @abc.abstractmethod
    def predict_with_bounds(self, X):
        pass

    def to_dict(self) -> dict[str, Any]:
        msg = f"Method to_dict for {_get_name_from_object(self)} not implemented"
        raise NotImplementedError(msg)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> None:
        msg = f"Method to_dict for {_get_name_from_type(cls)} not implemented"
        raise NotImplementedError(msg)


class _EnsembleBoundsEstimator(BoundsEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        for r in self.regressors:
            r.fit(X, y)

    def predict_with_bounds(self, X):
        ys = np.column_stack([r.predict(X) for r in self.regressors])
        y = np.mean(ys, axis=1)
        y_std = np.std(ys, ddof=1, axis=1)
        y_min = y - y_std
        y_max = y + y_std
        return y, y_min, y_max


class EnsembleRandomForestBoundsEstimator(_EnsembleBoundsEstimator):

    def __init__(self, n_regressors, **kwargs):
        super().__init__()
        main_seed = 14840861314273139720968056470480592917

        self.n_regressors = n_regressors
        random_states = np.random.SeedSequence(main_seed).spawn(self.n_regressors)

        self.regressors = [
            sklearn.ensemble.RandomForestRegressor(
                **kwargs, random_state=rs.generate_state(1)[0]
            )
            for rs in random_states
        ]
        assert len(self.regressors) == self.n_regressors


class EnsembleExtraTreesBoundsEstimator(_EnsembleBoundsEstimator):

    def __init__(self, n_regressors, **kwargs):
        super().__init__()
        main_seed = 14840861314273139720968056470480592917

        self.n_regressors = n_regressors
        random_states = np.random.SeedSequence(main_seed).spawn(self.n_regressors)

        self.regressors = [
            sklearn.ensemble.ExtraTreesRegressor(
                **kwargs, random_state=rs.generate_state(1)[0]
            )
            for rs in random_states
        ]
        assert len(self.regressors) == self.n_regressors


class EnsembleGradientBoostingBoundsEstimator(_EnsembleBoundsEstimator):

    def __init__(self, n_regressors, **kwargs):
        super().__init__()
        main_seed = 14840861314273139720968056470480592917

        self.n_regressors = n_regressors
        random_states = np.random.SeedSequence(main_seed).spawn(self.n_regressors)

        self.regressors = [
            sklearn.ensemble.GradientBoostingRegressor(
                **kwargs, random_state=rs.generate_state(1)[0]
            )
            for rs in random_states
        ]
        assert len(self.regressors) == self.n_regressors


class QuantileGradientBoostingBoundsEstimator(BoundsEstimator):

    def __init__(
        self,
        q_min,
        q_max,
        **kwargs,
    ):
        super().__init__()
        self.regressor_median = sklearn.ensemble.GradientBoostingRegressor(
            loss="quantile", alpha=0.5, **kwargs
        )
        self.regressor_min = sklearn.ensemble.GradientBoostingRegressor(
            loss="quantile", alpha=q_min, **kwargs
        )
        self.regressor_max = sklearn.ensemble.GradientBoostingRegressor(
            loss="quantile", alpha=q_max, **kwargs
        )

    def fit(self, X, y):
        self.regressor_median.fit(X, y)
        self.regressor_min.fit(X, y)
        self.regressor_max.fit(X, y)

    def predict_with_bounds(self, X):
        y = self.regressor_median.predict(X)
        y_min = self.regressor_min.predict(X)
        y_max = self.regressor_max.predict(X)
        return y, y_min, y_max

    def to_dict(self):
        v_ptblop, v_ptblopgen = utils.get_versions()
        return {
            "type": type(self).__name__,
            "regressor_median": _sklearn_to_str(self.regressor_median),
            "regressor_min": _sklearn_to_str(self.regressor_min),
            "regressor_max": _sklearn_to_str(self.regressor_max),
            "ptblop_version": v_ptblop,
            "ptblopgen_version": v_ptblopgen,
        }

    @classmethod
    def from_dict(cls, d) -> "QuantileGradientBoostingBoundsEstimator":
        reg = cls()
        reg.regressor_median = _str_to_sklearn(d["regressor_median"])
        reg.regressor_min = _str_to_sklearn(d["regressor_min"])
        reg.regressor_max = _str_to_sklearn(d["regressor_max"])
        return reg


class QuantileLinearEstimator(BoundsEstimator):

    def __init__(self, **kwargs):
        super().__init__()
        self.regressor_median = sklearn.linear_model.QuantileRegressor(
            quantile=0.5, **kwargs
        )
        self.regressor_min = sklearn.linear_model.QuantileRegressor(
            quantile=0.2, **kwargs
        )
        self.regressor_max = sklearn.linear_model.QuantileRegressor(
            quantile=0.8, **kwargs
        )

    def fit(self, X, y):
        self.regressor_median.fit(X, y)
        self.regressor_min.fit(X, y)
        self.regressor_max.fit(X, y)

    def predict_with_bounds(self, X):
        y = self.regressor_median.predict(X)
        y_min = self.regressor_min.predict(X)
        y_max = self.regressor_max.predict(X)
        return y, y_min, y_max


class QuantileTabPFNEstimator(BoundsEstimator):

    def __init__(self, q_min, q_max, **kwargs):
        self.q_min = q_min
        self.q_max = q_max
        self.regressor = tabpfn.TabPFNRegressor(**kwargs)

    def fit(self, X, y) -> None:
        self.regressor.fit(X, y)

    def predict_with_bounds(self, X):
        y, y_min, y_max = self.regressor.predict(
            X, output_type="quantiles", quantiles=[0.5, self.q_min, self.q_max]
        )
        return y, y_min, y_max


def save_regressor(fname, regressor: BoundsEstimator):
    with _open(fname, "wt") as f:
        d = regressor.to_dict()
        json.dump(d, f)


def load_regressor(fname):
    with _open(fname, "rt") as f:
        d = json.load(f)
    d_type = d["type"]
    if d_type == "QuantileGradientBoostingBoundsEstimator":
        reg = QuantileGradientBoostingBoundsEstimator.from_dict(d)
        return reg
    else:
        msg = f"Deserialization for {d_type} not implemented"
        raise ValueError(msg)
