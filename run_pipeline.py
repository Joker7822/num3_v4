#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numbers3 Orchestrator（範囲対応・cloudpickle版 / フルコード）

機能:
- 動的importで `numbers3_predictor.py` を読み込み
- モデルロード or 初回学習（cloudpickle直列化）
- 予測 → 保存（Numbers3_predictions.csv 追記・重複置換）→ Git push
- 実績があれば評価（evaluation_result.csv に追記・重複置換）
- 再学習 → モデル push
- 上記を日付ごとに実行（--start-date ～ --end-date）もしくは単一日（--target-date）

使い方:
  単一日:  python run_pipeline.py --repeat 1 --target-date 2025-08-08
  範囲   : python run_pipeline.py --start-date 1994-10-07 --end-date 2025-08-08

注意:
- 大量日付に対しては push 回数が増えます。リポジトリの運用方針に合わせてCI制限に注意してください。
"""
import argparse
import datetime as dt
import os
import sys
import traceback
import cloudpickle as cp
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Any

# ==== 動的 import（スペースや括弧を含むファイル名にも対応） ====
import importlib.util

DEFAULT_MODULE_PATH = Path("numbers3_predictor.py")
DEFAULT_DATA_CSV = Path("numbers3.csv")
DEFAULT_PRED_CSV = Path("Numbers3_predictions.csv")
DEFAULT_EVAL_CSV = Path("evaluation_result.csv")
DEFAULT_MODEL_PATH = Path("models/numbers3_model.pkl")

def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("numbers3_predictor_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def parse_number_string(x: Any) -> Optional[List[int]]:
    """ '1,2,3' / '[1, 2, 3]' / list -> [1,2,3] """
    if isinstance(x, list):
        try:
            return [int(v) for v in x][:3] if len(x) >= 3 else None
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip().replace("，", ",").replace("・", ",")
        s = s.strip("[](){}()")
        parts = [p for p in s.replace(" ", "").split(",") if p != ""]
        try:
            vals = [int(p) for p in parts]
            return vals[:3] if len(vals) >= 3 else None
        except Exception:
            return None
    return None

def load_numbers3(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} が見つかりません")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "抽せん日" not in df.columns or "本数字" not in df.columns:
        raise ValueError("numbers3.csv に '抽せん日' と '本数字' 列が必要です")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df["本数字"] = df["本数字"].apply(parse_number_string)
    df = df.dropna(subset=["抽せん日", "本数字"]).copy()
    df = df[df["本数字"].apply(lambda x: isinstance(x, list) and len(x) == 3)]
    return df

def get_actual_for_date(df: pd.DataFrame, date: dt.date) -> Optional[List[int]]:
    mask = df["抽せん日"].dt.date == date
    if not mask.any():
        return None
    row = df.loc[mask].sort_values("抽せん日").iloc[-1]
    return list(map(int, row["本数字"]))

def normalize_predictions(preds: Any) -> List[Tuple[List[int], float]]:
    """
    予測の形を [(numbers(list[int]), confidence(float)), ...] に揃える。
    - [{'numbers':[1,2,3], 'confidence':0.7, ...}, ...] → OK
    - {'numbers':[1,2,3], 'confidence':0.7} → 単一予測として扱う
    - [([1,2,3], 0.7), ...] → そのまま
    - [[1,2,3], [4,5,6], ...] → 信頼度は 0.5 で補完
    - "[1,2,3]" / "1,2,3" / "(1,2,3)" → 文字列も解釈
    - [1,2,3]（整数配列1件のみ）→ 単一予測として扱う
    """
    out: List[Tuple[List[int], float]] = []
    if preds is None:
        return out

    # 単一dict
    if isinstance(preds, dict):
        try:
            if "numbers" in preds:
                nums = [int(x) for x in preds.get("numbers", [])][:3]
                if len(nums) == 3:
                    conf = float(preds.get("confidence", 0.5))
                    out.append((nums, conf))
                    return out
        except Exception:
            pass

    # 文字列全体が一件の予測
    if isinstance(preds, str):
        nums = parse_number_string(preds)
        if nums and len(nums) == 3:
            out.append((nums, 0.5))
        return out

    # リスト/タプルが「一件の数字配列」の場合
    if isinstance(preds, (list, tuple)) and len(preds) >= 3 and all(isinstance(x, (int, str)) for x in preds):
        try:
            nums = [int(x) for x in preds][:3]
            if len(nums) == 3:
                out.append((nums, 0.5))
                return out
        except Exception:
            pass

    # 通常は「反復可能な複数予測」
    try:
        for p in preds:
            if isinstance(p, dict) and "numbers" in p:
                nums = [int(x) for x in p.get("numbers", [])][:3]
                conf = float(p.get("confidence", 0.5))
                if len(nums) == 3:
                    out.append((nums, conf))
            elif isinstance(p, (list, tuple)):
                if len(p) == 2 and isinstance(p[0], (list, tuple)):
                    nums = [int(x) for x in p[0]][:3]
                    conf = float(p[1])
                    if len(nums) == 3:
                        out.append((nums, conf))
                else:
                    nums = [int(x) for x in p][:3]
                    if len(nums) == 3:
                        out.append((nums, 0.5))
            elif isinstance(p, str):
                nums = parse_number_string(p)
                if nums and len(nums) == 3:
                    out.append((nums, 0.5))
    except Exception:
        # どの形式にも合致しない場合は空のまま返す
        pass
    return out

def generate_default_prediction(target_date: dt.date) -> List[int]:
    """
    予測が取得できなかった場合のフォールバック。
    日付から決定的に[0-9]の3桁を生成（乱数未使用）。
    """
    s = int(target_date.strftime('%Y%m%d'))
    a = (s * 1103515245 + 12345) & 0x7fffffff
    b = (a * 1103515245 + 54321) & 0x7fffffff
    c = (b * 1103515245 + 99991) & 0x7fffffff
    return [a % 10, b % 10, c % 10]

def save_eval_row(eval_csv: Path, date: dt.date, first_pred: List[int], grade: str):
    row = {
        "抽せん日": pd.to_datetime(date).strftime("%Y-%m-%d"),
        "予測1": str(first_pred),
        "等級": grade,
    }
    df_new = pd.DataFrame([row])
    if eval_csv.exists():
        try:
            df_old = pd.read_csv(eval_csv, encoding="utf-8-sig")
            df_old = df_old[df_old["抽せん日"] != row["抽せん日"]]
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            print(f"[WARN] {eval_csv} 読み込み失敗: {e} → 新規作成")
    df_new.to_csv(eval_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 評価結果を {eval_csv} に保存しました")

def daterange_list(start: dt.date, end: dt.date) -> List[dt.date]:
    if start > end:
        start, end = end, start
    days = (end - start).days
    return [start + dt.timedelta(days=i) for i in range(days + 1)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1, help="（単一日モードのみ）繰り返し回数")
    ap.add_argument("--module-path", type=str, default=str(DEFAULT_MODULE_PATH), help="numbers3_predictorファイルパス")
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA_CSV), help="numbers3.csv パス")
    ap.add_argument("--pred-csv", type=str, default=str(DEFAULT_PRED_CSV), help="Numbers3_predictions.csv パス")
    ap.add_argument("--eval-csv", type=str, default=str(DEFAULT_EVAL_CSV), help="evaluation_result.csv パス")
    ap.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="モデル保存先")
    ap.add_argument("--target-date", type=str, default=None, help="単一の予測対象日 (YYYY-MM-DD)")
    ap.add_argument("--start-date", type=str, default=None, help="範囲の開始日 (YYYY-MM-DD)")
    ap.add_argument("--end-date", type=str, default=None, help="範囲の終了日 (YYYY-MM-DD)")
    args = ap.parse_args()

    module_path = Path(args.module_path)
    data_csv = Path(args.data)
    pred_csv = Path(args.pred_csv)
    eval_csv = Path(args.eval_csv)
    model_path = Path(args.model_path)
    ensure_dir(model_path)

    # ロード
    mod = load_module(module_path)
    # cloudpickle 復元時のモジュール解決を安定化
    sys.modules.setdefault("numbers3_predictor_module", mod)
    # 必須関数/クラス
    LotoPredictor = getattr(mod, "LotoPredictor")
    save_predictions_to_csv = getattr(mod, "save_predictions_to_csv")
    evaluate_predictions = getattr(mod, "evaluate_predictions")
    git_commit_and_push = getattr(mod, "git_commit_and_push")
    classify_numbers3_prize = getattr(mod, "classify_numbers3_prize")

    # データ読み込み
    df = load_numbers3(data_csv)

    # 対象日リストを決める
    if args.start_date or args.end_date:
        start = pd.to_datetime(args.start_date if args.start_date else df["抽せん日"].min()).date()
        end = pd.to_datetime(args.end_date if args.end_date else dt.date.today()).date()
        date_list = daterange_list(start, end)
        print(f"[INFO] 範囲モード: {date_list[0]} ～ {date_list[-1]}（{len(date_list)}日）")
    else:
        # 単一日
        if args.target_date:
            date_list = [pd.to_datetime(args.target_date).date()]
        else:
            date_list = [dt.date.today()]
        print(f"[INFO] 単一日モード: {date_list[0]} / repeat={args.repeat}")

    # モデルのロード or 初回学習（cloudpickle）
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = cp.load(f)
            print(f"[INFO] 既存モデルをロード: {model_path}")
        except Exception as e:
            print(f"[WARN] モデルロード失敗 → 新規作成: {e}")
            model = LotoPredictor()
            # 最初の対象日で初回学習
            model.train_model(data=df, reference_date=date_list[0])
            with open(model_path, "wb") as f:
                cp.dump(model, f)
            git_commit_and_push(str(model_path), f"Add initial model {date_list[0]}")
    else:
        print("[INFO] 既存モデルなし → 新規学習を実行")
        model = LotoPredictor()
        model.train_model(data=df, reference_date=date_list[0])
        with open(model_path, "wb") as f:
            cp.dump(model, f)
        git_commit_and_push(str(model_path), f"Add initial model {date_list[0]}")

    # 実行ループ
    for target_date in date_list:
        repeats = args.repeat if len(date_list) == 1 else 1
        for i in range(1, repeats + 1):
            print(f"\n===== {target_date} / CYCLE {i}/{repeats} =====")
            try:
                # 1) 予測
                raw_preds = model.predict(latest_data=df, num_candidates=50)
                preds = normalize_predictions(raw_preds)
                if not preds:
                    print("[WARN] 予測結果が空 → フォールバック予測を使用")
                    fallback = generate_default_prediction(target_date)
                    preds = [(fallback, 0.3)]

                # 2) 保存 & push
                save_predictions_to_csv(
                    predictions=preds,
                    drawing_date=target_date,
                    filename=str(pred_csv),
                    model_name=f"model-{target_date}"
                )
                try:
                    git_commit_and_push(str(pred_csv), f"Auto update {Path(pred_csv).name} for {target_date} [skip ci]")
                except Exception as e:
                    print(f"[WARN] 予測CSVのpushに失敗: {e}")

                # 3) 評価（実績がある場合のみ）
                actual = get_actual_for_date(df, target_date)
                if actual is not None and len(actual) == 3:
                    results = evaluate_predictions(preds, actual)
                    grade = "はずれ"
                    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict) and "等級" in results[0]:
                        grade = results[0]["等級"]
                    elif isinstance(results, dict) and "等級" in results:
                        grade = results["等級"]
                    save_eval_row(eval_csv, target_date, preds[0][0], grade)
                else:
                    print("[INFO] 実績が未確定のため評価はスキップ")

                # 4) 再学習 → モデル保存 & push（cloudpickle）
                model.train_model(data=df, reference_date=target_date)
                with open(model_path, "wb") as f:
                    cp.dump(model, f)
                try:
                    git_commit_and_push(str(model_path), f"Update model after retrain {target_date}")
                except Exception as e:
                    print(f"[WARN] モデルpushに失敗: {e}")

            except Exception as e:
                print(f"[ERROR] {target_date} / CYCLE {i} で失敗: {e}\n{traceback.format_exc()}")
                # 次の日付へ継続

    print("\n[INFO] 完了")

if __name__ == "__main__":
    main()
