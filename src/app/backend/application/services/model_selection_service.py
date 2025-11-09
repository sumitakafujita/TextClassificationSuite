"""Optuna を用いたモデル選定ロジック."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import optuna
import pandas as pd
from sklearn import model_selection
from sklearn.base import BaseEstimator

from app.backend.config import BackendConfig
from app.backend.domain.entities import ModelSelectionResult
from app.backend.infrastructure.model_specs import CANDIDATE_SPECS


class ModelSelectionService:
    """候補モデルを Optuna でフィットし、最良モデルを決定するサービス."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config

    def _objective(
        self,
        trial: optuna.Trial,
        spec_name: str,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        spec = CANDIDATE_SPECS[spec_name]
        params = spec.search_space(trial)
        estimator = spec.builder(params, self._config.modeling.random_seed)
        cv = model_selection.StratifiedKFold(
            n_splits=self._config.modeling.cv_splits,
            shuffle=True,
            random_state=self._config.modeling.random_seed,
        )
        scores = model_selection.cross_val_score(
            estimator,
            features,
            labels,
            cv=cv,
            scoring="f1_macro",
            n_jobs=None,
        )
        return float(scores.mean())

    def _trial_count(self, name: str) -> int:
        modeling = self._config.modeling
        if name == "logistic_regression":
            return modeling.logistic_trials
        if name == "bagging_log_regression":
            return modeling.bagging_trials
        if name == "lightgbm_classifier":
            return modeling.lightgbm_trials
        if name == "mlp_classifier":
            return modeling.mlp_trials
        if name == "linear_svm":
            return modeling.linear_svm_trials
        return modeling.logistic_trials

    def run(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        candidate_names: List[str] | None = None,
    ) -> ModelSelectionResult:
        """全候補について Optuna チューニングを行い、最良モデルを返す."""
        summaries: List[Dict[str, float]] = []
        estimators: Dict[str, BaseEstimator] = {}
        cv_rows: List[Dict[str, float]] = []

        names = candidate_names or self._config.candidate_models
        for name in names:
            if name not in CANDIDATE_SPECS:
                continue
            spec = CANDIDATE_SPECS[name]
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self._config.modeling.random_seed))
            study.optimize(
                lambda trial, spec_name=name: self._objective(trial, spec_name, features, labels),
                n_trials=self._trial_count(name),
                show_progress_bar=False,
            )
            best_params = study.best_params
            best_score = study.best_value
            estimator = spec.builder(best_params, self._config.modeling.random_seed)
            estimator.fit(features, labels)
            estimators[name] = estimator
            summaries.append({"name": name, "best_macro_f1": best_score, **best_params})
            cv_rows.append({"name": name, "macro_f1": best_score})

        best = max(summaries, key=lambda row: row["best_macro_f1"])
        return ModelSelectionResult(
            best_model_name=best["name"],
            best_estimator=estimators[best["name"]],
            cv_summary=pd.DataFrame(cv_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True),
            study_summaries={row["name"]: row for row in summaries},
        )
