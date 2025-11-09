"""データ準備と埋め込み生成を行うサービス."""

from __future__ import annotations

from app.backend.config import BackendConfig
from app.backend.domain.entities import DatasetArtifacts, EmbeddingArtifacts
from app.backend.infrastructure.data_loader import PolarsDataLoader
from app.backend.infrastructure.embeddings import EmbeddingService


class DataPreparationService:
    """データセット読み込みから埋め込み生成までを担当."""

    def __init__(self, config: BackendConfig) -> None:
        """設定を受け取り、ローダと埋め込みサービスを初期化."""
        self._config = config
        self._loader = PolarsDataLoader(
            csv_path=config.data.data_path,
            text_column=config.data.text_column,
            label_column=config.data.label_column,
        )
        self._embedding_service = EmbeddingService(config.embedding)

    def load_dataset(self) -> DatasetArtifacts:
        """CSV を読み込み、ラベルをエンコードした結果を返す."""
        return self._loader.load()

    def from_dataframe(self, records: pd.DataFrame, text_column: str, label_column: str) -> DatasetArtifacts:
        """任意の DataFrame と列名を受け取り、データセットアーティファクトを生成."""
        required_cols = {text_column, label_column}
        missing = required_cols - set(records.columns)
        if missing:
            raise ValueError(f"指定されたカラムが見つかりません: {missing}")
        pdf = records.copy()
        encoder = LabelEncoder()
        pdf["category_id"] = encoder.fit_transform(pdf[label_column])
        labels = pdf["category_id"].values
        lazy = pl.from_pandas(pdf).lazy()
        return DatasetArtifacts(
            lazy_frame=lazy,
            records=pdf,
            labels=labels,
            label_encoder=encoder,
            text_column=text_column,
            label_column=label_column,
        )

    def build_embeddings(self, dataset: DatasetArtifacts) -> EmbeddingArtifacts:
        """本文とカテゴリ名を埋め込み、類似度特徴量も付与する."""
        return self._embedding_service.build_artifacts(
            texts=dataset.records[dataset.text_column],
            category_labels=dataset.label_encoder.classes_,
        )
