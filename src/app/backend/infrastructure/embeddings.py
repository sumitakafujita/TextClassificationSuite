"""SentenceTransformer を使った埋め込み生成ロジック."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from app.backend.config import EmbeddingConfig
from app.backend.domain.entities import EmbeddingArtifacts


class EmbeddingService:
    """テキストおよびカテゴリ名の埋め込み生成器."""

    def __init__(self, config: EmbeddingConfig) -> None:
        """埋め込みモデル設定を受け取り、SentenceTransformer を初期化."""
        self._config = config
        self._model = SentenceTransformer(config.model_name)

    def _normalize_text(self, text: str) -> str:
        """E5 系モデル向けに instruction プレフィックスと改行除去を行う."""
        clean = text.strip().replace("\n", " ")
        return f"{self._config.instruction}{clean}"

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        """テキスト群を埋め込みベクトルへ変換."""
        prepared = [self._normalize_text(text) for text in texts]
        vectors = self._model.encode(
            prepared,
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def build_artifacts(self, texts: Iterable[str], category_labels: Iterable[str]) -> EmbeddingArtifacts:
        """本文とカテゴリ名の両方を埋め込み、類似度特徴量を付与する."""
        text_vectors = self.encode_texts(texts)
        category_vectors = self.encode_texts(category_labels)
        similarities = text_vectors @ category_vectors.T
        augmented = np.hstack([text_vectors, similarities])
        return EmbeddingArtifacts(
            text_embeddings=text_vectors,
            augmented_embeddings=augmented,
            category_embeddings=category_vectors,
        )
