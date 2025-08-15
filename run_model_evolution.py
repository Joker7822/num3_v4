import os
import subprocess
import pandas as pd
from typing import Optional
from numbers3_predictor import (
    bulk_predict_all_past_draws,
    LotoPredictor,
    preprocess_data,
    set_global_seed
)

PRED_FILE = "Numbers3_predictions.csv"
ACTUAL_FILE = "numbers3.csv"
MODEL_PATH = "numbers3_model.pkl"  # 上書き保存

def _read_max_date_from_csv(path: str, col: str) -> Optional[pd.Timestamp]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path)
        if col not in df.columns or df.empty:
            return None
        df[col] = pd.to_datetime(df[col], errors="coerce")
        max_dt = df[col].max()
        return pd.to_datetime(max_dt) if pd.notna(max_dt) else None
    except Exception as e:
        print(f"[WARN] CSV読み込みに失敗: {path} → {e}")
        return None

def latest_actual_draw_date() -> Optional[pd.Timestamp]:
    return _read_max_date_from_csv(ACTUAL_FILE, "抽せん日")

def latest_predicted_draw_date() -> Optional[pd.Timestamp]:
    return _read_max_date_from_csv(PRED_FILE, "抽せん日")

def retrain_model_and_reset_predictions():
    """モデル再学習(上書き保存)→予測CSV削除→第1回目から全件再予測→即GIT PUSH"""
    print("[MODE] 予測は最新までカバー済み → 再学習＆リセット再予測を実行")

    # --- モデル再学習（上書き保存） ---
    set_global_seed(42)
    df = pd.read_csv(ACTUAL_FILE)
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")

    # 入力次元の見積り
    X, _, _ = preprocess_data(df)
    input_size = X.shape[1] if X is not None else 10

    predictor = LotoPredictor(input_size=input_size, hidden_size=128)
    predictor.train_model(df)

    # 上書き保存（joblibを使ってシンプルに保存）
    try:
        import joblib
        joblib.dump(predictor, MODEL_PATH)
        print(f"[SAVE] モデルを上書き保存しました: {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] モデル保存に失敗 (継続します): {e}")

    # --- 予測CSV削除 ---
    if os.path.exists(PRED_FILE):
        try:
            os.remove(PRED_FILE)
            print(f"[CLEAN] {PRED_FILE} を削除しました")
        except Exception as e:
            print(f"[WARN] 予測ファイル削除に失敗: {e}")

    # --- 第1回目からの全件再予測 ---
    print("[REBUILD] 過去全抽せん回を対象に再予測を実行します...")
    bulk_predict_all_past_draws()
    print("[DONE] 再学習＆リセット再予測が完了しました。")

    # --- 再学習直後にGIT PUSH ---
    try:
        print("[GIT] コミット＆PUSHを実行します...")
        subprocess.run(["git", "config", "--global", "user.name", "github-actions"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "github-actions@github.com"], check=True)
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", "Auto retrain Numbers3 model and predictions [skip ci]"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("[GIT] PUSH完了")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] GIT PUSHに失敗しました: {e}")

def continue_predictions():
    """まだカバーされていないので、予測を継続（不足分を含め全体を再構築）"""
    print("[MODE] 予測が最新まで未カバー → 予測を継続します")
    bulk_predict_all_past_draws()
    print("[DONE] 予測の継続処理が完了しました。")

def main():
    actual_max = latest_actual_draw_date()
    pred_max = latest_predicted_draw_date()

    print(f"[INFO] 最新の抽せん日 (numbers3.csv): {actual_max}")
    print(f"[INFO] 予測ファイルの最新日付 (Numbers3_predictions.csv): {pred_max}")

    if actual_max is not None and pred_max is not None and pred_max >= actual_max:
        retrain_model_and_reset_predictions()
    else:
        continue_predictions()

if __name__ == "__main__":
    main()
