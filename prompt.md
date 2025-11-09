# ゴール
`data/text_classification_samples_200.csv` の `text` 列から `category_label` を高精度に予測できるベースラインを用意し、後続タスクでも使い回せる形で PoC をまとめる。

# タスク
1. CSV を Polars の `LazyFrame` で読み込み、最小限の確認（件数、列情報、サンプル）を行う。必要になるまで `collect` はしない。
2. Materialize した後は `LabelEncoder` などで目的変数を数値化し、EDA 用の要約を 2〜3 個（件数、カテゴリ分布など）出す。
3. `SentenceTransformer`（`intfloat/multilingual-e5-small`）で `text` を instruction prefix 付きの埋め込みベクトルへ変換する。関数化してバッチサイズを引数で変えられるようにする。
4. 3 モデル以上で学習・推論パイプラインを用意する。必須: SVM（`LinearSVC`）、ロジスティック回帰、LightGBM。追加で Bagging（例: BaggingClassifier + LogisticRegression）や Stacking/Voting などバギング以外のアンサンブルも含めて比較すること。SVM/LogReg/Bagging は `Pipeline` + `StandardScaler` で統一。
5. `train_test_split`（stratify あり, test_size=0.2）でデータを分割し、各モデルについて accuracy / macro F1 / `classification_report` を取得する共通関数を用意する。クロスバリデーション（5-fold）での平均精度も記録して、ランキング形式で表示する。
6. 主要モデル（少なくともロジスティック回帰と LightGBM）のハイパーパラメータ調整は Optuna で行い、ベストスコアとパラメータをログとして残す。

# 出力
1. `PoC.ipynb` に処理をセル単位で整理する。各コードセルは先頭で「Purpose: ...」コメントを書き、後続タスクでも再利用できるように関数／モデルレジストリ化する。
2. Notebook には `load_lazy_dataset`、`materialize_dataset`、`build_embedder`、`embed_texts`、`train_test_split_embeddings`、`evaluate_models`、`summarize_results` などの関数を含める。
3. 評価結果（表形式）と最良モデルの `classification_report` を Notebook に表示し、LightGBM や Bagging の精度を比較できるようにする。この `prompt.md` を最新手順に合わせて維持する。
