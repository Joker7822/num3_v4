import os
import pandas as pd
from datetime import datetime
from numbers3_predictor import (
    LotoPredictor,
    save_predictions_to_csv,
    evaluate_and_summarize_predictions
)

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
