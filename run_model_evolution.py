
import os
import sys
import pandas as pd
from datetime import datetime

# Import pipeline utilities from numbers3_predictor
from numbers3_predictor import (
    bulk_predict_all_past_draws,
    evaluate_and_summarize_predictions,
    LotoPredictor,
    preprocess_data,
    set_global_seed,
)

PRED_FILE = "Numbers3_predictions.csv"
ACTUAL_FILE = "numbers3.csv"

def read_latest_draw_date(csv_path: str) -> pd.Timestamp:
    df = pd.read_csv(csv_path)
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    latest = df["抽せん日"].max()
    if pd.isna(latest):
        raise ValueError("numbers3.csv に有効な抽せん日が見つかりません。")
    return latest.normalize()

def read_latest_pred_date(pred_path: str) -> pd.Timestamp | None:
    if not os.path.exists(pred_path) or os.path.getsize(pred_path) == 0:
        return None
    try:
        df = pd.read_csv(pred_path)
        if "抽せん日" not in df.columns or df.empty:
            return None
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
        latest = df["抽せん日"].max()
        if pd.isna(latest):
            return None
        return latest.normalize()
    except Exception:
        return None

def continue_predictions():
    print("[PREDICT] 予測が最新ではないため、欠落分の予測を実行します...")
    # 現状の実装では、過去全抽せん回を安全に再生成します（重複は上書き処理側で排除）
    bulk_predict_all_past_draws()
    print("[PREDICT] 予測処理が完了しました。")

def retrain_models():
    print("[RETRAIN] 予測が最新なので、評価→再学習を行います...")
    try:
        evaluate_and_summarize_predictions(
            pred_file=PRED_FILE,
            actual_file=ACTUAL_FILE,
            output_csv="evaluation_result.csv",
            output_txt="evaluation_summary.txt",
        )
        print("[RETRAIN] 評価完了: evaluation_result.csv / evaluation_summary.txt")
    except Exception as e:
        print(f"[RETRAIN][WARN] 評価に失敗: {e}（再学習は継続）")

    try:
        set_global_seed(42)
        df = pd.read_csv(ACTUAL_FILE)
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
        X, _, _ = preprocess_data(df)
        input_size = X.shape[1] if X is not None else 10
        predictor = LotoPredictor(input_size=input_size, hidden_size=128)
        predictor.train_model(df)
        print("[RETRAIN] 再学習完了")
    except Exception as e:
        print(f"[RETRAIN][ERROR] 再学習に失敗: {e}")

def main():
    print("=== Numbers3 Orchestrator ===")
    try:
        latest_draw = read_latest_draw_date(ACTUAL_FILE)
        print(f"[INFO] 最新の抽せん日: {latest_draw.date()}")
    except Exception as e:
        print(f"[ERROR] 抽せん日取得に失敗: {e}")
        sys.exit(1)

    latest_pred = read_latest_pred_date(PRED_FILE)
    if latest_pred is None:
        print("[INFO] 予測ファイルがない/空/不正のため、予測を開始します。")
        continue_predictions()
        return

    print(f"[INFO] 予測ファイル内の最新抽せん日: {latest_pred.date()}")

    if latest_pred >= latest_draw:
        retrain_models()
    else:
        continue_predictions()

if __name__ == "__main__":
    main()
