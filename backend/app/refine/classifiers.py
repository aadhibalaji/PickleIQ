"""Stage B — live/dead classification, two interchangeable backends.

Both backends implement :class:`LiveClassifier`: ``predict_proba(features)``
returns P(live) per timestep. The rules classifier is the zero-training
baseline and permanent fallback; the learned classifier is XGBoost when the
package is importable, else a pure-numpy L2 logistic regression trained on
the same feature matrix (Stage E provides labels).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import RefineConfig
from .features import Features

log = logging.getLogger("refine.classifiers")

try:
    import xgboost as _xgb
except ImportError:                                   # pragma: no cover
    _xgb = None


class LiveClassifier:
    """Common interface: per-timestep P(live) from a Features object."""

    def predict_proba(self, feats: Features) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, names: list[str]) -> None:
        raise NotImplementedError("this backend is not trainable")


class RulesClassifier(LiveClassifier):
    """Hand-written weighted rules squashed through a sigmoid.

    Positive evidence: recent net crossings, ball speed, reactivity, players
    in position. Negative: ball parked near one player, ball long out of
    court. Weights/bias live in the config, so the threshold sweep can tune
    them like any other knob.
    """

    def __init__(self, cfg: RefineConfig):
        self.cfg = cfg

    def predict_proba(self, feats: Features) -> np.ndarray:
        z = np.full(len(feats.t), self.cfg.rules_bias)
        for name, w in self.cfg.rules_weights.items():
            z += w * feats.col(name)
        return 1.0 / (1.0 + np.exp(-z))


class _NumpyLogistic:
    """Minimal L2 logistic regression (fallback when xgboost is missing)."""

    def __init__(self, l2: float = 1e-2, lr: float = 0.3, iters: int = 800):
        self.l2, self.lr, self.iters = l2, lr, iters
        self.w: np.ndarray | None = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        self.w = np.zeros(d)
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))
            g = X.T @ (p - y) / n + self.l2 * self.w
            self.w -= self.lr * g
            self.b -= self.lr * float(np.mean(p - y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))

    def to_dict(self) -> dict:
        return {"kind": "logistic", "w": self.w.tolist(), "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> "_NumpyLogistic":
        m = cls()
        m.w = np.asarray(d["w"])
        m.b = float(d["b"])
        return m


class XGBoostClassifier(LiveClassifier):
    """Learned P(live). XGBoost when available, numpy logistic otherwise.

    The saved artifact records the feature-name order it was trained with and
    re-orders columns at prediction time, so feature additions don't silently
    scramble a stale model.
    """

    def __init__(self, cfg: RefineConfig):
        self.cfg = cfg
        self.names: list[str] | None = None
        self.model = None
        self._is_xgb = False

    # -- training ----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, names: list[str]) -> None:
        self.names = list(names)
        if _xgb is not None:
            # native booster API — the sklearn wrapper requires scikit-learn
            p = dict(self.cfg.xgb_params)
            rounds = int(p.pop("n_estimators", 200))
            p.setdefault("eta", p.pop("learning_rate", 0.08))
            p.update(objective="binary:logistic", eval_metric="logloss")
            dtrain = _xgb.DMatrix(X, label=y, feature_names=self.names)
            self.model = _xgb.train(p, dtrain, num_boost_round=rounds)
            self._is_xgb = True
        else:
            log.warning("xgboost unavailable — training numpy logistic fallback")
            self.model = _NumpyLogistic()
            self.model.fit(X, y)
            self._is_xgb = False

    # -- inference -----------------------------------------------------------
    def _matrix(self, feats: Features) -> np.ndarray:
        assert self.names is not None, "model not loaded/trained"
        return np.column_stack([feats.col(n) for n in self.names])

    def predict_proba(self, feats: Features) -> np.ndarray:
        return self.predict_matrix(self._matrix(feats))

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        if self._is_xgb:
            return self.model.predict(_xgb.DMatrix(X, feature_names=self.names))
        return self.model.predict_proba(X)

    # -- persistence -----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._is_xgb:
            self.model.save_model(path)
            meta = {"kind": "xgboost", "names": self.names}
        else:
            meta = {"kind": "logistic", "names": self.names,
                    "model": self.model.to_dict()}
        path.with_suffix(".meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str | Path, cfg: RefineConfig) -> "XGBoostClassifier":
        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        obj = cls(cfg)
        obj.names = meta["names"]
        if meta["kind"] == "xgboost":
            if _xgb is None:
                raise RuntimeError("model was trained with xgboost, which is "
                                   "not importable here")
            obj.model = _xgb.Booster()
            obj.model.load_model(str(path))
            obj._is_xgb = True
        else:
            obj.model = _NumpyLogistic.from_dict(meta["model"])
            obj._is_xgb = False
        return obj


def make_classifier(cfg: RefineConfig) -> LiveClassifier:
    """Backend per config; a missing/broken model artifact falls back to
    rules with a warning rather than failing the whole pass."""
    if cfg.classifier == "xgboost":
        try:
            return XGBoostClassifier.load(cfg.model_path, cfg)
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            log.warning("could not load model %s (%s) — falling back to rules",
                        cfg.model_path, e)
    return RulesClassifier(cfg)
