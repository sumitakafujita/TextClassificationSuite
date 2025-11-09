# テキスト分類実験ログ

## データセット
- 参照ファイル: `data/text_classification_samples_200.csv`
- 200 件の日本語問い合わせテキスト (カテゴリ `C1`〜`C4`)
- クラス分布: クレーム 55 / 質問 55 / お礼・ポジティブ 46 / 要望 44 とおおむね均衡

## 埋め込み戦略
1. **Baseline (PoC.ipynb 内の初期セル)**: `sentence-transformers/all-MiniLM-L6-v2` を instruction なしで利用
2. **Improved (PoC.ipynb 最新セル)**: `intfloat/multilingual-e5-small` に `query: ` のプレフィックスを付与し正規化済みベクトルを取得
3. **カテゴリ類似度特徴量**: カテゴリ名を埋め込み、テキストとのコサイン類似度を 4 次元の追加特徴として結合 (384 → 388 次元)

## モデル群
- Linear SVM (`LinearSVC` + StandardScaler)
- Logistic Regression (StandardScaler + `LogisticRegression`)
- MLP Classifier (隠れ層 384, ReLU, 1500 epoch)
- LightGBM (マルチクラス)
- Bagging(Logistic Regression)
- Ensembles: Stacking(LogReg + LightGBM) / soft Voting(LogReg + LightGBM)

## 時系列の知見
### 1. PoC.ipynb (MiniLM 埋め込み)
- 分割: 80/20 Stratified split
- 最高性能: Logistic Regression (Accuracy 0.72 / Macro-F1 0.73)
- Linear SVM は 0.68, Gradient Boosting は 0.53 と伸び悩み

### 2. PoC (E5 埋め込み, チューニング前)
- 5-fold CV:
  - Logistic Regression: Accuracy 0.960 ± 0.025 / Macro-F1 0.959 ± 0.026
  - Bagging(LogReg): Accuracy 0.965 ± 0.025 / Macro-F1 0.965 ± 0.027
  - LightGBM: Accuracy 0.900 ± 0.069 / Macro-F1 0.902 ± 0.066
- Hold-out (80/20):
  - Logistic Regression: Accuracy 0.95 / Macro-F1 0.95
  - Bagging(LogReg): Accuracy 0.93 / Macro-F1 0.925
  - LightGBM: Accuracy 0.88 / Macro-F1 0.879

### 3. Optuna チューニング (LogReg & LightGBM) + Stacking/Voting
- Optuna 設定: 5-fold、TPESampler、LogReg 25 trials / LightGBM 30 trials
- ベスト LogReg: `C≈0.043`, `fit_intercept=False`, `class_weight='balanced'`, `tol≈6.9e-03`, Macro-F1 ≈ 0.975
- ベスト LightGBM: `num_leaves=53`, `max_depth=10`, `learning_rate≈0.024`, `feature_fraction≈0.924`, `bagging_fraction≈0.601`, `bagging_freq=4`, `min_child_samples=11`, `lambda_l1≈0.002`, `lambda_l2≈0.026`, Macro-F1 ≈ 0.940
- Stacking / Voting いずれも Hold-out で Accuracy 0.95 / Macro-F1 0.95

### 4. カテゴリ類似度特徴量
- カテゴリ埋め込みとのコサイン類似度を4特徴として追加
- CV リーダーボードは Logistic & Bagging が 0.96 付近、LightGBM は 0.89 程度
- Hold-out も 0.95 前後を維持

### 5. 確信度 & 類似度診断
- Optuna 調整済み LogReg を全件学習し、確信度 (予測確率) を取得。閾値 0.85 で ~10件の低確信度サンプルが抽出される
- 同一予測ラベル内でコサイン類似度 >= 0.92 の近傍をリスト化し、重複問い合わせやテンプレ改善を洗い出しやすくした
- `PoC.ipynb` では ID 5,11 などが近傍として抽出され、運用ツール連携の解像度が上がった

## 推奨方針
1. **プロダクション候補**: E5 + 類似度特徴量 + Optuna 調整 LogReg (または Bagging) … 軽量で安定して Macro-F1 >= 0.95
2. **アンサンブル案**: LogReg + LightGBM の Stacking / Voting で頑健性を確保
3. **追加アイデア**
   - `joblib` 等でベストモデルを永続化し API 連携
   - カテゴリごとのプロトタイプ埋め込み (平均ベクトル) や説明文を使った類似度強化
   - Platt / Isotonic などで確信度のキャリブレーション

PoC の最新コード／出力は `PoC.ipynb` に統合済み。要件の変遷は `prompt.md` に記録しています。
