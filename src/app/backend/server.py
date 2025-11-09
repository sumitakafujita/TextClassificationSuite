"""FastAPI エントリーポイント."""

from __future__ import annotations

import io
import pickle
import uuid
from typing import Any, Dict

import pandas as pd
import polars as pl
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.backend.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    DatasetUploadResponse,
    ModelUploadResponse,
    PretrainedClassificationRequest,
)
from app.backend.pipeline import ClassificationBackend

app = FastAPI(title="ShipAI Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS: Dict[str, pd.DataFrame] = {}
MODELS: Dict[str, Any] = {}
PIPELINE = ClassificationBackend()


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    """稼働確認用の簡易エンドポイント."""
    return {"status": "ok"}


@app.post("/api/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    """CSV を受け取り、サーバー側でデータセットIDを発行する."""
    content = await file.read()
    try:
        table = pl.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"CSV の読み込みに失敗しました: {exc}") from exc
    records = table.to_pandas()
    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = records
    return DatasetUploadResponse(
        dataset_id=dataset_id,
        columns=list(records.columns),
        row_count=len(records),
    )


@app.post("/api/models/upload", response_model=ModelUploadResponse)
async def upload_model(file: UploadFile = File(...)) -> ModelUploadResponse:
    """pickle 化された sklearn パイプラインを登録する."""
    content = await file.read()
    try:
        estimator = pickle.loads(content)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"pickle の読み込みに失敗しました: {exc}") from exc
    if not hasattr(estimator, "predict") or not hasattr(estimator, "predict_proba"):
        raise HTTPException(
            status_code=400,
            detail="predict / predict_proba を備えた sklearn 互換モデルをアップロードしてください。",
        )
    model_id = str(uuid.uuid4())
    MODELS[model_id] = estimator
    return ModelUploadResponse(model_id=model_id, model_type=estimator.__class__.__name__)


@app.post("/api/classify/auto", response_model=ClassificationResponse)
async def classify_auto(payload: ClassificationRequest) -> ClassificationResponse:
    """アップロード済みデータセットから自動的に最適なモデルを探索する."""
    records = _get_dataset(payload.dataset_id)
    result = PIPELINE.run_with_dataframe(
        records=records,
        text_column=payload.text_column,
        label_column=payload.label_column,
        candidate_models=payload.candidate_models,
    )
    return _serialize_result(result)


@app.post("/api/classify/pretrained", response_model=ClassificationResponse)
async def classify_with_model(payload: PretrainedClassificationRequest) -> ClassificationResponse:
    """アップロード済みモデルで推論結果を取得する."""
    records = _get_dataset(payload.dataset_id)
    estimator = _get_model(payload.model_id)
    result = PIPELINE.run_with_pretrained(
        estimator=estimator,
        records=records,
        text_column=payload.text_column,
        label_column=payload.label_column,
    )
    return _serialize_result(result)


def _get_dataset(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="指定されたデータセットIDが存在しません。")
    return DATASETS[dataset_id]


def _get_model(model_id: str) -> Any:
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="指定されたモデルIDが存在しません。")
    return MODELS[model_id]


def _serialize_df(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


def _serialize_result(result: Dict[str, Any]) -> ClassificationResponse:
    return ClassificationResponse(
        best_model=result["best_model"],
        cv_summary=_serialize_df(result.get("cv_summary")),
        study_summaries=result.get("study_summaries", {}),
        confidence_table=_serialize_df(result.get("confidence_table")),
        low_confidence=_serialize_df(result.get("low_confidence")),
        similar_groups=_serialize_df(result.get("similar_groups")),
        embedding_points=_serialize_df(result.get("embedding_points")),
    )
