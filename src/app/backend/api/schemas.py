"""FastAPI 用のスキーマ定義."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetUploadResponse(BaseModel):
    """データセットアップロード結果."""

    dataset_id: str = Field(..., description="サーバー側で管理するデータセットID")
    columns: List[str] = Field(..., description="CSV に含まれるカラム一覧")
    row_count: int = Field(..., description="レコード件数")


class ModelUploadResponse(BaseModel):
    """学習済みモデルのアップロード結果."""

    model_id: str = Field(..., description="登録されたモデルID")
    model_type: str = Field(..., description="モデルのクラス名")


class ClassificationRequest(BaseModel):
    """自動モデリングを実行するためのリクエスト."""

    dataset_id: str
    text_column: str
    label_column: str
    candidate_models: Optional[List[str]] = Field(
        default=None, description="限定的に試したいモデル名のリスト"
    )


class PretrainedClassificationRequest(ClassificationRequest):
    """アップロード済みモデルで推論するためのリクエスト."""

    model_id: str


class ClassificationResponse(BaseModel):
    """推論結果と各種診断情報."""

    best_model: str
    cv_summary: List[Dict[str, Any]]
    study_summaries: Dict[str, Dict[str, Any]]
    confidence_table: List[Dict[str, Any]]
    low_confidence: List[Dict[str, Any]]
    similar_groups: List[Dict[str, Any]]
    embedding_points: List[Dict[str, Any]]
