"""バックエンド処理全体をまとめたパイプライン."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from sklearn.decomposition import PCA

from app.backend.application.services.data_service import DataPreparationService
from app.backend.application.services.diagnostic_service import DiagnosticService
from app.backend.application.services.model_selection_service import ModelSelectionService
from app.backend.config import BackendConfig
from app.backend.domain.entities import DatasetArtifacts, EmbeddingArtifacts


class ClassificationBackend:
    """PoC ノートブックで検証した流れをコード化したバックエンド."""

    def __init__(self, config: BackendConfig | None = None) -> None:
        self._config = config or BackendConfig()
        self._data_service = DataPreparationService(self._config)
        self._model_service = ModelSelectionService(self._config)
        self._diagnostic_service = DiagnosticService(self._config)

    def run(self, candidate_models: Optional[List[str]] = None) -> Dict[str, object]:
        """既定の CSV を対象に全処理を実行."""
        dataset = self._data_service.load_dataset()
        return self._run_with_dataset(dataset, candidate_models)

    def run_with_dataframe(
        self,
        records: pd.DataFrame,
        text_column: str,
        label_column: str,
        candidate_models: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """任意の DataFrame とカラム指定で処理を実行."""
        dataset = self._data_service.from_dataframe(records, text_column, label_column)
        return self._run_with_dataset(dataset, candidate_models)

    def run_with_pretrained(
        self,
        estimator,
        records: pd.DataFrame,
        text_column: str,
        label_column: str,
    ) -> Dict[str, object]:
        """アップロード済みモデルを使って診断情報を生成."""
        dataset = self._data_service.from_dataframe(records, text_column, label_column)
        embeddings = self._data_service.build_embeddings(dataset)
        return self._build_response(
            model_name="uploaded_model",
            estimator=estimator,
            dataset=dataset,
            embeddings=embeddings,
            selection_summary=None,
        )

    def _run_with_dataset(
        self,
        dataset: DatasetArtifacts,
        candidate_models: Optional[List[str]],
    ) -> Dict[str, object]:
        embeddings = self._data_service.build_embeddings(dataset)
        selection = self._model_service.run(
            features=embeddings.augmented_embeddings,
            labels=dataset.labels,
            candidate_names=candidate_models,
        )
        return self._build_response(
            model_name=selection.best_model_name,
            estimator=selection.best_estimator,
            dataset=dataset,
            embeddings=embeddings,
            selection_summary=selection,
        )

    def _build_response(
        self,
        model_name: str,
        estimator,
        dataset: DatasetArtifacts,
        embeddings: EmbeddingArtifacts,
        selection_summary,
    ) -> Dict[str, object]:
        confidence = self._diagnostic_service.build_confidence_report(
            estimator=estimator,
            features=embeddings.augmented_embeddings,
            records=dataset.records,
            label_encoder=dataset.label_encoder,
            text_column=dataset.text_column,
            label_column=dataset.label_column,
        )
        similar_groups = self._diagnostic_service.find_similar_within_pred(
            embeddings=embeddings.text_embeddings,
            record_ids=dataset.records["id"],
            texts=dataset.records[dataset.text_column],
            predicted_labels=confidence.detail["pred_label"],
        )
        projection = self._project_embeddings(
            embeddings.text_embeddings,
            dataset.records["id"],
            confidence.detail["pred_label"],
            confidence.detail["confidence"],
            dataset.records[dataset.label_column],
        )
        return {
            "best_model": model_name,
            "cv_summary": selection_summary.cv_summary if selection_summary else pd.DataFrame(),
            "study_summaries": selection_summary.study_summaries if selection_summary else {},
            "confidence_table": confidence.detail.sort_values("confidence"),
            "low_confidence": confidence.low_confidence.sort_values("confidence"),
            "similar_groups": pd.json_normalize([group.__dict__ for group in similar_groups], sep="."),
            "embedding_points": projection,
        }

    def _project_embeddings(
        self,
        text_embeddings,
        record_ids,
        predicted_labels,
        confidences,
        true_labels,
    ) -> pd.DataFrame:
        """テキスト埋め込みを 3 次元に射影し、可視化用データを生成."""
        reducer = PCA(n_components=3, random_state=self._config.modeling.random_seed)
        coords = reducer.fit_transform(text_embeddings)
        return pd.DataFrame(
            {
                "id": record_ids,
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
                "pred_label": predicted_labels,
                "true_label": true_labels,
                "confidence": confidences,
            }
        )
