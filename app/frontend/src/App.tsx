import {
  Box,
  Button,
  Card,
  CardContent,
  Container,
  Grid,
  LinearProgress,
  MenuItem,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import Autocomplete from "@mui/material/Autocomplete";
import axios from "axios";
import Plot from "react-plotly.js";
import { ChangeEvent, useMemo, useState } from "react";

type ClassificationRow = Record<string, unknown>;

interface ClassificationResult {
  best_model: string;
  cv_summary: ClassificationRow[];
  study_summaries: Record<string, ClassificationRow>;
  low_confidence: ClassificationRow[];
  confidence_table: ClassificationRow[];
  similar_groups: ClassificationRow[];
  embedding_points: ClassificationRow[];
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";

const candidateOptions = [
  { label: "Logistic Regression", value: "logistic_regression" },
  { label: "Bagging Logistic", value: "bagging_log_regression" },
  { label: "MLP Classifier", value: "mlp_classifier" },
  { label: "Linear SVM", value: "linear_svm" },
  { label: "LightGBM", value: "lightgbm_classifier" },
];

const App = () => {
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [datasetColumns, setDatasetColumns] = useState<string[]>([]);
  const [textColumn, setTextColumn] = useState("");
  const [labelColumn, setLabelColumn] = useState("");
  const [candidateModels, setCandidateModels] = useState<string[]>([]);
  const [modelId, setModelId] = useState<string | null>(null);
  const [results, setResults] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  const handleDatasetUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files?.length) return;
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("file", file);
    setStatus("データセットをアップロード中...");
    try {
      const { data } = await axios.post(`${API_BASE}/datasets/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setDatasetId(data.dataset_id);
      setDatasetColumns(data.columns);
      setTextColumn("");
      setLabelColumn("");
      setStatus(`データセットを登録しました (ID: ${data.dataset_id})`);
    } catch (error) {
      console.error(error);
      setStatus("アップロードに失敗しました。");
    }
  };

  const handleModelUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files?.length) return;
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("file", file);
    setStatus("モデルをアップロード中...");
    try {
      const { data } = await axios.post(`${API_BASE}/models/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setModelId(data.model_id);
      setStatus(`モデルを登録しました (ID: ${data.model_id})`);
    } catch (error) {
      console.error(error);
      setStatus("モデルのアップロードに失敗しました。");
    }
  };

  const runAutoPipeline = async () => {
    if (!datasetId || !textColumn || !labelColumn) {
      setStatus("データセットIDとカラムの指定が必要です。");
      return;
    }
    setLoading(true);
    setStatus("Auto ML を実行中...");
    try {
      const payload = {
        dataset_id: datasetId,
        text_column: textColumn,
        label_column: labelColumn,
        candidate_models: candidateModels.length ? candidateModels : undefined,
      };
      const { data } = await axios.post(`${API_BASE}/classify/auto`, payload);
      setResults(data);
      setStatus(`最良モデル: ${data.best_model}`);
    } catch (error) {
      console.error(error);
      setStatus("推論に失敗しました。");
    } finally {
      setLoading(false);
    }
  };

  const runPretrained = async () => {
    if (!datasetId || !modelId || !textColumn || !labelColumn) {
      setStatus("データ・カラム・モデルIDを確認してください。");
      return;
    }
    setLoading(true);
    setStatus("アップロード済みモデルで推論中...");
    try {
      const payload = {
        dataset_id: datasetId,
        text_column: textColumn,
        label_column: labelColumn,
        model_id: modelId,
      };
      const { data } = await axios.post(`${API_BASE}/classify/pretrained`, payload);
      setResults(data);
      setStatus(`カスタムモデルの診断が完了しました。`);
    } catch (error) {
      console.error(error);
      setStatus("カスタムモデルでの推論に失敗しました。");
    } finally {
      setLoading(false);
    }
  };

  const plotData = useMemo(() => {
    if (!results?.embedding_points?.length) return null;
    const x = results.embedding_points.map((row) => Number(row.x));
    const y = results.embedding_points.map((row) => Number(row.y));
    const z = results.embedding_points.map((row) => Number(row.z));
    const labels = results.embedding_points.map(
      (row) => `ID: ${row.id} / pred: ${row.pred_label} / conf: ${(Number(row.confidence) * 100).toFixed(1)}%`,
    );
    const colors = results.embedding_points.map((row) => Number(row.confidence));
    return [
      {
        type: "scatter3d",
        mode: "markers",
        x,
        y,
        z,
        text: labels,
        marker: {
          size: 6,
          color: colors,
          colorscale: "Viridis",
          opacity: 0.9,
        },
      },
    ];
  }, [results]);

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h4" fontWeight="bold" gutterBottom>
        ShipAI Text Classification Studio
      </Typography>
      <Typography color="text.secondary" gutterBottom>
        データセットをアップロードし、テキスト列とカテゴリ列を選んでワンクリックでモデリング。確信度の低いサンプルや類似問い合わせ、3D プロットでの直感的な分析をサポートします。
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                データセット
              </Typography>
              <Stack spacing={2}>
                <Button variant="contained" component="label">
                  CSV を選択
                  <input hidden type="file" accept=".csv" onChange={handleDatasetUpload} />
                </Button>
                {datasetColumns.length > 0 && (
                  <>
                    <TextField
                      select
                      label="テキスト列"
                      value={textColumn}
                      onChange={(e) => setTextColumn(e.target.value)}
                      size="small"
                    >
                      {datasetColumns.map((col) => (
                        <MenuItem key={col} value={col}>
                          {col}
                        </MenuItem>
                      ))}
                    </TextField>
                    <TextField
                      select
                      label="カテゴリ列"
                      value={labelColumn}
                      onChange={(e) => setLabelColumn(e.target.value)}
                      size="small"
                    >
                      {datasetColumns.map((col) => (
                        <MenuItem key={col} value={col}>
                          {col}
                        </MenuItem>
                      ))}
                    </TextField>
                    <Autocomplete
                      multiple
                      options={candidateOptions}
                      value={candidateOptions.filter((option) => candidateModels.includes(option.value))}
                      onChange={(_, value) => setCandidateModels(value.map((item) => item.value))}
                      renderInput={(params) => <TextField {...params} label="試すモデル (任意)" size="small" />}
                    />
                  </>
                )}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                モデル
              </Typography>
              <Stack spacing={2}>
                <Button variant="outlined" component="label">
                  pickle をインポート
                  <input hidden type="file" accept=".pkl,.pickle" onChange={handleModelUpload} />
                </Button>
                {modelId && <Typography fontSize={14}>利用中のモデルID: {modelId}</Typography>}
                <Button variant="contained" disabled={loading} onClick={runAutoPipeline}>
                  Auto ML を実行
                </Button>
                <Button variant="contained" color="secondary" disabled={loading || !modelId} onClick={runPretrained}>
                  アップロードモデルで推論
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ステータス
              </Typography>
              {loading && <LinearProgress sx={{ mb: 2 }} />}
              <Typography color="text.secondary">{status ?? "操作を開始してください。"}</Typography>
              {results && (
                <Box mt={2}>
                  <Typography variant="subtitle1">ベストモデル: {results.best_model}</Typography>
                  {results.cv_summary.length > 0 && (
                    <Typography variant="body2" color="text.secondary">
                      CV Macro-F1: {(results.cv_summary[0].macro_f1 as number).toFixed?.(3) ?? results.cv_summary[0].macro_f1}
                    </Typography>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {results && (
        <Box mt={4}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    確信度の低いサンプル
                  </Typography>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>ID</TableCell>
                        <TableCell>予測ラベル</TableCell>
                        <TableCell>確信度</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {results.low_confidence.slice(0, 10).map((row) => (
                        <TableRow key={String(row.id)}>
                          <TableCell>{row.id as string}</TableCell>
                          <TableCell>{row.pred_label as string}</TableCell>
                          <TableCell>{((row.confidence as number) * 100).toFixed(1)}%</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    類似問い合わせグループ
                  </Typography>
                  <Stack spacing={1} maxHeight={320} sx={{ overflowY: "auto" }}>
                    {results.similar_groups.slice(0, 6).map((group) => (
                      <Box key={`${group.root_id}-${group.predicted_label}`} border={1} borderRadius={2} p={1}>
                        <Typography fontWeight="bold">
                          #{group.root_id} ({group.predicted_label})
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {group.root_text}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          近傍: {(group.neighbors as unknown[]).length}件
                        </Typography>
                      </Box>
                    ))}
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Box mt={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  3D 埋め込みプロット
                </Typography>
                {plotData ? (
                  <Plot
                    data={plotData}
                    layout={{
                      autosize: true,
                      height: 520,
                      margin: { l: 0, r: 0, b: 0, t: 0 },
                      scene: { xaxis: { title: "PC1" }, yaxis: { title: "PC2" }, zaxis: { title: "PC3" } },
                    }}
                    config={{ responsive: true, displaylogo: false }}
                    style={{ width: "100%", height: "100%" }}
                  />
                ) : (
                  <Typography color="text.secondary">プロット対象のデータがありません。</Typography>
                )}
              </CardContent>
            </Card>
          </Box>
        </Box>
      )}
    </Container>
  );
};

export default App;
