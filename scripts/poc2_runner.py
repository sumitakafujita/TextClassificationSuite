"""Execution helper for PoC2.ipynb with embedding + TF-IDF fusion."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import os

HF_CACHE_DIR = r"C:\Users\fujita096\.cache\huggingface"
HF_HUB_SUBDIR = rf"{HF_CACHE_DIR}\hub"
HF_MODEL_SNAPSHOT_DIR = Path(HF_HUB_SUBDIR) / "models--intfloat--multilingual-e5-small" / "snapshots"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", HF_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HUB_SUBDIR)
from sentence_transformers import SentenceTransformer
from sklearn import metrics, model_selection, preprocessing
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
DATA_PATH = Path("data/data.csv")
EMBED_MODEL = "intfloat/multilingual-e5-small"
E5_INSTRUCTION = "query: "
TFIDF_MAX_FEATURES = 60000
SVD_COMPONENTS = 192
HOLDOUT_SIZE = 0.2


def normalize_text(title: str, body: str) -> str:
    title = title or ""
    body = body or ""
    merged = f"{title.strip()}\n{body.strip()}".strip()
    return unicodedata.normalize("NFKC", merged)


def build_embeddings(embedder: SentenceTransformer, texts: Iterable[str]) -> np.ndarray:
    prepared = [f"{E5_INSTRUCTION}{text.replace(chr(10), ' ')}" for text in texts]
    return embedder.encode(
        prepared,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def resolve_model_source() -> str:
    if HF_MODEL_SNAPSHOT_DIR.exists():
        snapshots = sorted(
            [p for p in HF_MODEL_SNAPSHOT_DIR.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snap in snapshots:
            if (snap / "config.json").exists():
                return str(snap)
    return EMBED_MODEL


def build_feature_matrix(
    df: pd.DataFrame,
    encoder: preprocessing.LabelEncoder,
) -> Tuple[np.ndarray, Dict[str, object]]:
    texts = [normalize_text(t, b) for t, b in zip(df["アイデアタイトル"], df["text"])]
    model_source = resolve_model_source()
    embedder = SentenceTransformer(model_source, cache_folder=HF_CACHE_DIR)
    doc_embeddings = build_embeddings(embedder, texts)
    category_embeddings = build_embeddings(embedder, encoder.classes_)
    similarity_features = doc_embeddings @ category_embeddings.T

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=TFIDF_MAX_FEATURES,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_SEED)
    tfidf_dense = svd.fit_transform(tfidf_matrix)
    tfidf_scaled = StandardScaler().fit_transform(tfidf_dense)

    length_features = np.vstack(
        [
            df["text"].str.len().fillna(0).to_numpy(),
            df["アイデアタイトル"].str.len().fillna(0).to_numpy(),
            df["text"].str.count("。").fillna(0).to_numpy(),
        ]
    ).T
    length_scaled = StandardScaler().fit_transform(length_features)

    fused = np.hstack([doc_embeddings, similarity_features, tfidf_scaled, length_scaled])
    fused_scaled = StandardScaler().fit_transform(fused)
    artifacts = {
        "vectorizer": vectorizer,
        "svd": svd,
    }
    return fused_scaled, artifacts


def build_model_registry(random_state: int = RANDOM_SEED):
    def make_logistic(C: float) -> LogisticRegression:
        return LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=C,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            multi_class="multinomial",
            random_state=random_state,
        )

    return {
        "log_reg_balanced": lambda: make_logistic(1.75),
    }


def cross_validate_models(
    registry,
    features: np.ndarray,
    labels: np.ndarray,
    cv_splits: int = 3,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    splitter = model_selection.StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=seed
    )
    rows: List[Dict[str, float]] = []
    for name, builder in registry.items():
        estimator = builder()
        scores = model_selection.cross_validate(
            estimator,
            features,
            labels,
            cv=splitter,
            scoring=["accuracy", "f1_macro"],
            n_jobs=None,
        )
        rows.append(
            {
                "name": name,
                "cv_accuracy_mean": scores["test_accuracy"].mean(),
                "cv_accuracy_std": scores["test_accuracy"].std(),
                "cv_macro_f1_mean": scores["test_f1_macro"].mean(),
                "cv_macro_f1_std": scores["test_f1_macro"].std(),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("cv_macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_holdout(
    estimator,
    features: np.ndarray,
    labels: np.ndarray,
    label_names: Iterable[str],
    test_size: float = HOLDOUT_SIZE,
    seed: int = RANDOM_SEED,
) -> Dict[str, object]:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        features,
        labels,
        stratify=labels,
        test_size=test_size,
        random_state=seed,
    )
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    return {
        "accuracy": metrics.accuracy_score(y_test, preds),
        "macro_f1": metrics.f1_score(y_test, preds, average="macro"),
        "report": classification_report(y_test, preds, target_names=list(label_names)),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    encoder = preprocessing.LabelEncoder()
    df["category_id"] = encoder.fit_transform(df["category"])
    fused_features, artifacts = build_feature_matrix(df, encoder)
    registry = build_model_registry()
    cv_results = cross_validate_models(registry, fused_features, df["category_id"].values)
    best_name = cv_results.loc[0, "name"]
    best_model = clone(registry[best_name]())
    holdout = evaluate_holdout(
        best_model, fused_features, df["category_id"].values, encoder.classes_
    )
    results = {
        "cv_results": cv_results.to_dict(orient="records"),
        "best_model": best_name,
        "holdout": holdout,
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
