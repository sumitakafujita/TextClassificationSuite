"""データアクセスを担当するインフラ層の実装."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder

from app.backend.domain.entities import DatasetArtifacts


class PolarsDataLoader:
    """Polars を用いて CSV を読み込むデータローダ."""

    def __init__(self, csv_path: Path, text_column: str, label_column: str) -> None:
        """CSV パスおよびカラム情報を保持."""
        self._csv_path = csv_path
        self._text_column = text_column
        self._label_column = label_column

    def load(self) -> DatasetArtifacts:
        """LazyFrame と Pandas DataFrame を返却しつつ、ラベルをエンコードする."""
        lazy = pl.scan_csv(self._csv_path)
        pdf = lazy.collect().to_pandas()
        encoder = LabelEncoder()
        pdf["category_id"] = encoder.fit_transform(pdf[self._label_column])
        labels = pdf["category_id"].values
        return DatasetArtifacts(
            lazy_frame=lazy,
            records=pdf,
            labels=labels,
            label_encoder=encoder,
            text_column=self._text_column,
            label_column=self._label_column,
        )
