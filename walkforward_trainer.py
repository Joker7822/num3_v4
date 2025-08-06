import os
import pandas as pd
from datetime import datetime
from numbers3_predictor import (
    LotoPredictor,
    save_predictions_to_csv,
    evaluate_and_summarize_predictions
)
import torch
import onnxruntime
import numpy as np
from autogluon.tabular import TabularPredictor

# === 設定 ===
MIN_TRAIN_SIZE = 30
PRED_FILE = "Numbers3_predictions.csv"
EVAL_FILE = "evaluation_result.csv"
BACKUP_DIR = "walkforward_eval"
os.makedirs(BACKUP_DIR, exist_ok=True)

# === データ読み込みと整備 ===
data = pd.read_csv("numbers3.csv")
data["抽せん日"] = pd.to_datetime(data["抽せん日"], errors="coerce")
data = data.dropna(subset=["抽せん日", "本数字"]).sort_values("抽せん日").reset_index(drop=True)

# === モデル初期化 ===
predictor = LotoPredictor(input_size=10, hidden_size=128)

# === ウォークフォワードループ ===
for i in range(MIN_TRAIN_SIZE, len(data)):
    train_df = data.iloc[:i].copy()
    test_df = data.iloc[[i]].copy()
    draw_date = test_df.iloc[0]["抽せん日"]

    print(f"\n===== {i+1} 回目 ({draw_date.date()}) の予測開始 =====")

    try:
        predictor.train_model(train_df, reference_date=draw_date)
        predictions, _ = predictor.predict(test_df)
        save_predictions_to_csv(predictions, draw_date, filename=PRED_FILE, model_name="WalkF")

        # === モデル保存（LSTM / AutoGluon） ===
        today = datetime.now().strftime("%Y%m%d")

        # LSTM → ONNX保存
        lstm_dir = f"models/lstm/{today}/"
        os.makedirs(lstm_dir, exist_ok=True)
        onnx_path = os.path.join(lstm_dir, "lstm_model.onnx")
        dummy_input = torch.randn(1, 1, predictor.input_size)
        torch.onnx.export(
            predictor.lstm_model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=12
        )
        print(f"[INFO] LSTMモデルをONNX形式で保存: {onnx_path}")

        # AutoGluon保存
        ag_dir = f"models/autogluon/{today}/"
        os.makedirs(ag_dir, exist_ok=True)
        for idx, model in enumerate(predictor.regression_models):
            if model:
                model_path = os.path.join(ag_dir, f"digit{idx}")
                model.save(model_path)
                print(f"[INFO] AutoGluonモデル {idx} を保存: {model_path}")

    except Exception as e:
        print(f"[ERROR] {draw_date.date()} の処理中にエラー: {e}")
        continue

# === 最終評価 ===
evaluate_and_summarize_predictions(pred_file=PRED_FILE, output_csv=EVAL_FILE)

# === 評価結果のバックアップ ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_csv = os.path.join(BACKUP_DIR, f"evaluation_result_{timestamp}.csv")
backup_pred = os.path.join(BACKUP_DIR, f"predictions_{timestamp}.csv")

if os.path.exists(EVAL_FILE):
    os.rename(EVAL_FILE, backup_csv)
    print(f"[INFO] 評価結果をバックアップ保存: {backup_csv}")

if os.path.exists(PRED_FILE):
    os.rename(PRED_FILE, backup_pred)
    print(f"[INFO] 予測ログをバックアップ保存: {backup_pred}")

# === 保存済みモデルからの推論関数 ===
def predict_with_saved_models(lstm_path, autogluon_dir, input_data):
    """
    lstm_path: ONNXファイルへのパス
    autogluon_dir: 各桁のAutoGluonモデルが格納されたディレクトリ
    input_data: shape=(1, feature_dim) の np.array または pd.DataFrame
    """
    # LSTM推論
    try:
        sess = onnxruntime.InferenceSession(lstm_path)
        input_name = sess.get_inputs()[0].name
        lstm_out = sess.run(None, {input_name: input_data.reshape(1, 1, -1).astype(np.float32)})
        lstm_preds = [int(np.argmax(out)) for out in lstm_out[:3]]
    except Exception as e:
        print(f"[ERROR] LSTMモデル推論失敗: {e}")
        lstm_preds = [0, 0, 0]

    # AutoGluon推論
    try:
        if isinstance(input_data, np.ndarray):
            input_df = pd.DataFrame(input_data, columns=[f"f{i}" for i in range(input_data.shape[1])])
        else:
            input_df = input_data

        auto_preds = []
        for i in range(3):
            model_path = os.path.join(autogluon_dir, f"digit{i}")
            predictor = TabularPredictor.load(model_path)
            pred = predictor.predict(input_df).values[0]
            auto_preds.append(int(pred))
    except Exception as e:
        print(f"[ERROR] AutoGluonモデル推論失敗: {e}")
        auto_preds = [0, 0, 0]

    # 結果を平均的に統合
    final = [int(round((l + a) / 2)) for l, a in zip(lstm_preds, auto_preds)]
    print(f"[PREDICT] 統合予測結果: {final}")
    return final
