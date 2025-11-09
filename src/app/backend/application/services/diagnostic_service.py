"""確信度評価や類似テキスト探索を行うサービス."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity

from app.backend.config import BackendConfig
from app.backend.domain.entities import ConfidenceReport, SimilarityGroup


class DiagnosticService:
    """推論後の診断機能（確信度・類似度）を提供する."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config

    def build_confidence_report(
        self,
        estimator: BaseEstimator,
        features: np.ndarray,
        records: pd.DataFrame,
        label_encoder,
        text_column: str,
        label_column: str,
    ) -> ConfidenceReport:
        """確信度の高低を算出し、低確信度サンプルを抽出する."""
        probs = estimator.predict_proba(features)
        pred_ids = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        detail = pd.DataFrame(
            {
                "id": records["id"],
                "text": records[text_column],
                "true_label": records[label_column],
                "pred_label": label_encoder.inverse_transform(pred_ids),
                "confidence": confidences,
            }
        )
        detail["low_confidence"] = detail["confidence"] < self._config.modeling.low_confidence_threshold
        low_conf = detail[detail["low_confidence"]].sort_values("confidence")
        return ConfidenceReport(detail=detail, low_confidence=low_conf)

    def find_similar_within_pred(
        self,
        embeddings: np.ndarray,
        record_ids: Iterable[int],
        texts: Iterable[str],
        predicted_labels: Iterable[str],
    ) -> List[SimilarityGroup]:
        """同じ予測ラベル内でコサイン類似度が高い組み合わせを列挙する."""
        ids = np.asarray(list(record_ids))
        label_array = np.asarray(list(predicted_labels))
        text_array = np.asarray(list(texts))
        cosine = cosine_similarity(embeddings)
        groups: List[SimilarityGroup] = []
        for idx in range(len(embeddings)):
            same_mask = label_array == label_array[idx]
            candidate_indices = np.where(same_mask)[0]
            sims = cosine[idx, candidate_indices]
            neighbors = []
            for neighbor_idx, sim in zip(candidate_indices, sims):
                if neighbor_idx == idx or sim < self._config.modeling.similarity_threshold:
                    continue
                neighbors.append(
                    {
                        "neighbor_id": int(ids[neighbor_idx]),
                        "similarity": float(sim),
                        "neighbor_text": text_array[neighbor_idx],
                    }
                )
            if neighbors:
                neighbors = sorted(neighbors, key=lambda x: x["similarity"], reverse=True)[
                    : self._config.modeling.similarity_top_k
                ]
                groups.append(
                    SimilarityGroup(
                        root_id=int(ids[idx]),
                        root_text=text_array[idx],
                        predicted_label=label_array[idx],
                        neighbors=neighbors,
                    )
                )
        return groups
