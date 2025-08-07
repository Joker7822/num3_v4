import pandas as pd
import time
import os
from numbers3_predictor import (
    LotoPredictor,
    save_predictions_to_csv,
    calculate_next_draw_date,
    git_commit_and_push
)

def load_all_data(csv_path="numbers3.csv"):
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} が見つかりません")
        return None
    df = pd.read_csv(csv_path)
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    return df

def run_reinforcement_cycle(iterations=3):
    changes_made = False

    for i in range(iterations):
        print(f"\n===== 🔁 CYCLE {i+1} 開始 =====")

        # 1. データ読み込み
        data = load_all_data()
        if data is None or data.empty:
            print("[ERROR] 有効なデータが読み込めません。処理を中断します。")
            break

        try:
            # 2. モデル再学習
            predictor = LotoPredictor(input_size=20, hidden_size=128)
            predictor.train_model(data)

            # 3. 予測（未来リーク防止済み）
            predictions, _ = predictor.predict(data)
            draw_date = calculate_next_draw_date()
            save_predictions_to_csv(predictions, draw_date, model_name=f"RL_Cycle_{i+1}")

            # 4. モデルファイル存在確認
            model_path = "lstm_model.onnx"
            if os.path.exists(model_path):
                changes_made = True

        except Exception as e:
            print(f"[ERROR] CYCLE {i+1} にて例外が発生しました: {e}")
            continue

        # 5. 小休止（必要に応じて削減可能）
        time.sleep(1)

    print("=== ✅ 全ての強化ループ完了 ===")

    # 6. コミットとPush（1回だけ）
    if changes_made:
        print("[INFO] モデルまたは予測データに変更あり。Gitに反映します。")
        git_commit_and_push("lstm_model.onnx", "[AutoCycle] 自動強化学習モデル更新")
        git_commit_and_push("self_predictions.csv", "[AutoCycle] 自己予測CSV更新")
        git_commit_and_push("self_predictions_gen/", "[AutoCycle] 世代別自己予測の更新")
    else:
        print("[INFO] 変更がなかったため、Gitへのpushはスキップされました。")

if __name__ == "__main__":
    run_reinforcement_cycle(iterations=3)
