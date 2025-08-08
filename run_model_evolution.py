"""
このスクリプトは `numbers3_predictor.py` に含まれる機能を用いて、Numbers3の予測と学習のパイプラインを実行します。

元の `run_model_evolution.py` ではロト7の予測と評価を行っていましたが、
本ファイルではNumbers3に対応させるために以下の変更を加えています。

1. 予測ファイル名を `Numbers3_predictions.csv` に変更し、古いファイルを削除します。
2. `numbers3_predictor.bulk_predict_all_past_draws()` を呼び出して過去すべての抽せん回に対する予測を実施します。
   この関数は必要に応じてGPTやPPO、Diffusionなど複数のモデルを統合して予測し、
   予測結果を `Numbers3_predictions.csv` に書き出します。
3. 予測結果の評価として `evaluate_and_summarize_predictions` を呼び出し、
   `evaluation_result.csv` と `evaluation_summary.txt` を生成します。これは的中率や等級分布を確認するためのものです。
4. 新しい予測結果と評価を反映するために `LotoPredictor` を再学習します。
   Numbers3版の `LotoPredictor` は `train_model` の引数に精度評価データを取らないため、
   単に学習データフレームを渡します。評価結果は内部で利用される `evaluation_result.csv` から取り込まれます。

このスクリプトを実行すると、最新の予測ファイルを作成し、評価を行った上で学習モデルを強化します。
"""

import os
import pandas as pd
from numbers3_predictor import (
    bulk_predict_all_past_draws,
    evaluate_and_summarize_predictions,
    LotoPredictor,
    preprocess_data,
    set_global_seed,
)


def main():
    # === ステップ 1: 既存予測ファイルを削除 ===
    print("[STEP 1] 古い予測ファイルを削除")
    pred_file = "Numbers3_predictions.csv"
    if os.path.exists(pred_file):
        os.remove(pred_file)
        print(f"[INFO] {pred_file} を削除しました")

    # === ステップ 2: 過去全データで未来リーク防止予測 ===
    print("[STEP 2] すべての抽せん回について予測を実施中...")
    try:
        bulk_predict_all_past_draws()
    except Exception as e:
        print(f"[ERROR] 予測実行中にエラー: {e}")

    # === ステップ 3: 精度評価（等級・一致数など）===
    print("[STEP 3] 予測結果の精度を評価中...")
    try:
        evaluate_and_summarize_predictions(
            pred_file="Numbers3_predictions.csv",
            actual_file="numbers3.csv",
            output_csv="evaluation_result.csv",
            output_txt="evaluation_summary.txt",
        )
    except Exception as e:
        print(f"[ERROR] 精度評価中にエラー: {e}")

    # === ステップ 4: モデル再学習 ===
    print("[STEP 4] 予測精度を使ってモデルを再学習中...")
    try:
        set_global_seed(42)
        df = pd.read_csv("numbers3.csv")
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors='coerce')
        # 特徴量を作成して入力次元を決定
        X, _, _ = preprocess_data(df)
        input_size = X.shape[1] if X is not None else 10
        # Numbers3用LotoPredictorを初期化して学習
        predictor = LotoPredictor(input_size=input_size, hidden_size=128)
        predictor.train_model(df)
        print("[✅ 完了] モデルの強化学習が完了しました")
    except Exception as e:
        print(f"[ERROR] モデル再学習中にエラー: {e}")


if __name__ == "__main__":
    main()
