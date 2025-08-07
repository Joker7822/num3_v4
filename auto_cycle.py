import pandas as pd
import time
import os
from numbers3_predictor import LotoPredictor, save_predictions_to_csv, calculate_next_draw_date, git_commit_and_push

def load_all_data(csv_path="numbers3.csv"):
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} が見つかりません")
        return None
    df = pd.read_csv(csv_path)
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    return df

def run_reinforcement_cycle(iterations=5):
    for i in range(iterations):
        print(f"\n===== 🔁 CYCLE {i+1} 開始 =====")

        # 1. データ読み込み
        data = load_all_data()
        if data is None:
            break

        # 2. モデル再学習
        predictor = LotoPredictor(input_size=20, hidden_size=128)
        predictor.train_model(data)

        # 3. 予測を未来リーク無しで再実行
        predictions, _ = predictor.predict(data)
        draw_date = calculate_next_draw_date()
        save_predictions_to_csv(predictions, draw_date, model_name="LotoPredictor_RL")

        # 4. モデルの保存（ここでは例としてLSTMのみ）
        model_path = "lstm_model.onnx"
        if os.path.exists(model_path):
            git_commit_and_push(model_path, f"[AutoCycle] {i+1}回目の再学習モデル")

        # 5. 次のループまで小休止
        time.sleep(1)

    print("=== ✅ 全ての強化ループ完了 ===")

if __name__ == "__main__":
    run_reinforcement_cycle(iterations=3)
