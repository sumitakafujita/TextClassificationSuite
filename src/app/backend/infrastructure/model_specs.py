"""モデル候補の定義と Optuna で用いるサーチスペース."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import optuna
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class CandidateSpec:
    """候補モデルのビルド方法とサーチスペース."""

    name: str
    builder: Callable[[Dict[str, float], int], Pipeline]
    search_space: Callable[[optuna.Trial], Dict[str, float]]


def _build_logistic(params: Dict[str, float], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=params["C"],
                    fit_intercept=params["fit_intercept"],
                    class_weight=params["class_weight"],
                    tol=params["tol"],
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _space_logistic(trial: optuna.Trial) -> Dict[str, float]:
    return {
        "C": trial.suggest_float("C", 1e-2, 10.0, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
    }


def _build_bagging(params: Dict[str, float], seed: int) -> Pipeline:
    base = LogisticRegression(
        C=params["C"],
        fit_intercept=params["fit_intercept"],
        class_weight=params["class_weight"],
        tol=params["tol"],
        max_iter=5000,
        random_state=seed,
    )
    bagging = BaggingClassifier(
        estimator=base,
        n_estimators=int(params["n_estimators"]),
        max_samples=params["max_samples"],
        bootstrap=True,
        random_state=seed,
        n_jobs=None,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", bagging)])


def _space_bagging(trial: optuna.Trial) -> Dict[str, float]:
    return {
        "C": trial.suggest_float("C", 1e-2, 10.0, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 5, 25),
        "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
    }


def _build_lightgbm(params: Dict[str, float], seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=int(params["n_estimators"]),
        learning_rate=params["learning_rate"],
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        subsample=params["subsample"],
        subsample_freq=int(params["subsample_freq"]),
        colsample_bytree=params["colsample_bytree"],
        min_child_samples=int(params["min_child_samples"]),
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _space_lightgbm(trial: optuna.Trial) -> Dict[str, float]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _build_mlp(params: Dict[str, float], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(int(params["hidden_size"]),),
                    activation="relu",
                    alpha=params["alpha"],
                    learning_rate_init=params["learning_rate_init"],
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _space_mlp(trial: optuna.Trial) -> Dict[str, float]:
    return {
        "hidden_size": trial.suggest_int("hidden_size", 128, 512, step=32),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    }


def _build_linear_svm(params: Dict[str, float], seed: int) -> Pipeline:
    base = LinearSVC(
        C=params["C"],
        tol=params["tol"],
        class_weight=params["class_weight"],
        random_state=seed,
        max_iter=5000,
    )
    calibrated = CalibratedClassifierCV(base_estimator=base, cv=3, method="sigmoid")
    return Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])


def _space_linear_svm(trial: optuna.Trial) -> Dict[str, float]:
    return {
        "C": trial.suggest_float("C", 1e-2, 10.0, log=True),
        "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


CANDIDATE_SPECS = {
    "logistic_regression": CandidateSpec(
        name="logistic_regression",
        builder=_build_logistic,
        search_space=_space_logistic,
    ),
    "bagging_log_regression": CandidateSpec(
        name="bagging_log_regression",
        builder=_build_bagging,
        search_space=_space_bagging,
    ),
    "mlp_classifier": CandidateSpec(
        name="mlp_classifier",
        builder=_build_mlp,
        search_space=_space_mlp,
    ),
    "linear_svm": CandidateSpec(
        name="linear_svm",
        builder=_build_linear_svm,
        search_space=_space_linear_svm,
    ),
    "lightgbm_classifier": CandidateSpec(
        name="lightgbm_classifier",
        builder=_build_lightgbm,
        search_space=_space_lightgbm,
    ),
}
