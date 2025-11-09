## Text Classification Suite

もともとは ShipAI という案件向けに作った仕組みですが、実装自体は **どの案件でもテキスト列とカテゴリ列を指定するだけで再利用できる** 汎用的な構成になっています。  
Polars + SentenceTransformer + scikit-learn を基盤に、Optuna を用いた自動モデル選定、確信度解析、類似メッセージ集約までを一通り自動化し、FastAPI + React の UI から扱えるように構成しています。

### 主な機能
- **データ取込:** 任意の CSV をアップロードし、対象のテキスト列・カテゴリ列を選択するだけでモデリング開始。
- **自動モデリング:** Logistic / Bagging / MLP / Linear SVM / LightGBM の候補を Optuna でチューニングし、ベストモデルを選出。
- **カスタムモデル診断:** pickle 化された学習済みモデルをアップロードして同じ診断 (低確信度、類似グルーピング、3D 可視化) を実施。
- **確信度監視:** 低確信度の予測を閾値順に並べ、人手で優先確認できる。
- **類似グループ抽出:** 予測カテゴリごとに埋め込みのコサイン類似度が高い問い合わせをまとめて表示し、重複やテンプレ改善に役立てる。
- **3D 埋め込み可視化:** PCA で 3 次元に射影し、Plotly で視覚的にクラスタを確認。

---

## リポジトリ構成

```
src/app/backend/     # FastAPI サーバー本体とドメインロジック
app/frontend/        # Vite + React + MUI + Plotly のフロントエンド
PoC.ipynb            # Notebook ベースの検証ログ（最新ワークフロー）
EXPERIMENTS.md       # 実験の履歴と要約
```

バックエンドのコードはクリーンアーキテクチャ風に整理されており、`pipeline.ClassificationBackend` が Notebook で行っていた一連の処理 (データ読み込み -> 埋め込み -> モデリング -> 診断) を再利用可能なサービスとして提供します。

---

## セットアップ

### 1. Python 依存解決
```bash
uv sync
```

### 2. FastAPI バックエンド起動
```bash
uv run uvicorn app.backend.server:app --reload --app-dir src
# http://localhost:8000/api/health で疎通確認
```

### 3. フロントエンド起動
```bash
cd app/frontend
npm install
# バックエンドの URL を変える場合は環境変数をセット
# VITE_API_BASE_URL=http://localhost:8000/api npm run dev
npm run dev
```
ブラウザで `http://localhost:5173` (Vite の表示 URL) にアクセスすると UI が利用できます。

---

## API サマリ

| Method | Path | 説明 |
| --- | --- | --- |
| GET | `/api/health` | ヘルスチェック |
| POST | `/api/datasets/upload` | CSV アップロード。返値に dataset_id とカラム一覧 |
| POST | `/api/models/upload` | pickle モデルの登録。返値に model_id |
| POST | `/api/classify/auto` | Auto ML 実行 (dataset_id, text_column, label_column, candidate_models) |
| POST | `/api/classify/pretrained` | アップロード済みモデルで診断 (`model_id` 追加) |

レスポンスには最良モデル名、CV サマリ、study パラメータ、確信度テーブル、低確信度リスト、類似グルーピング、3D 座標が含まれます。

---

## フロントエンドの使い方
1. 「CSV を選択」からデータセットをアップロード。MUI のメニューからテキスト列とカテゴリ列を選びます。
2. 「試すモデル」で任意の候補 (LogReg / Bagging / MLP / Linear SVM / LightGBM) を制限できます。
3. Auto ML ボタンでモデリング開始。`status` カードに進捗とベストモデルが表示されます。
4. 低確信度テーブルで優先監視すべき記録を確認。類似グループカードで重複・テンプレ改善につながる塊を把握。
5. Plotly の 3D 表示でクラスタ構造を視覚的に確認。
6. 既に学習済みの pipeline/pickle がある場合は「pickle をインポート」でアップロードし、「アップロードモデルで推論」を実行すると同じ診断が得られます。

---

## ノートブックと実験ログ
- `PoC.ipynb` : E5 埋め込み、Optuna、類似度特徴量、確信度/類似グループ分析まで含む最新 PoC  
  (初期の MiniLM ベースラインも同ノートにまとめ済み)
- `EXPERIMENTS.md` : 全ステップの背景・結果を時系列で記録

---

## 発展アイデア
- 推論 API の非同期化・キュー化、結果キャッシュ
- 類似グループに対する自動タグ付けやクラスタ命名
- 確信度閾値に基づく自動ルーティング (例: 低確信度 -> 人手確認)
- Notebook 操作用に pipeline を CLI 化 (`python -m app.backend.cli …`)

ご不明点があれば Issue / Pull Request でお知らせください。別案件への横展開やカラム構成の違いがあっても、この README の手順でそのまま再利用できます。
