import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, time
from numbers3_predictor import (
    main_with_improved_predictions,
    evaluate_and_summarize_predictions
)

# ========= 自動予測実行のチェック =========
LOG_FILE = "last_prediction_log.txt"

def already_predicted_today():
    today_str = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            last_run = f.read().strip()
            return last_run == today_str
    return False

def mark_prediction_done():
    today_str = datetime.now().strftime("%Y-%m-%d")
    with open(LOG_FILE, "w") as f:
        f.write(today_str)

# ========= ページ設定・UI =========
st.set_page_config(page_title="Numbers3予測AI", layout="wide")
st.markdown("<h1 style='color:#FF4B4B;'>🎯 Numbers3 予測AI</h1>", unsafe_allow_html=True)

# 現在時刻の表示（ログ確認用）
now = datetime.now()
st.caption(f"🕒 現在時刻: {now.strftime('%Y-%m-%d %H:%M:%S')}")

# ========= メニュー =========
menu = st.sidebar.radio("📌 メニュー", [
    "🧠 最新予測表示", 
    "📊 予測評価", 
    "📉 予測分析グラフ", 
    "🧾 予測結果表示"
])

# ========= 自動予測実行ブロック =========
if now.weekday() < 5 and now.time() >= time(21, 0) and not already_predicted_today():
    with st.spinner("⏳ 平日21:00を過ぎたため、自動で予測を実行しています..."):
        try:
            main_with_improved_predictions()
            mark_prediction_done()
            st.success("✅ 本日の自動予測が完了しました")
        except Exception as e:
            st.error(f"❌ 自動予測中にエラーが発生しました: {e}")
else:
    if already_predicted_today():
        st.info("✅ 本日の予測はすでに実行済みです。")
    elif now.weekday() >= 5:
        st.info("📴 土日は自動予測を行いません。")
    elif now.time() < time(21, 0):
        st.info("⏳ 本日の自動予測は 21:00 以降に実行されます。")

# ========= 最新予測表示 =========
if menu == "🧠 最新予測表示":
    st.markdown("## 🧠 最新予測結果")
    with st.container():
        if os.path.exists("Numbers3_predictions.csv"):
            try:
                pred_df = pd.read_csv("Numbers3_predictions.csv")
                latest_row = pred_df.sort_values("抽せん日", ascending=False).iloc[0]

                st.success("✅ 最新予測が取得されました")

                st.markdown(f"""
                    <div style='padding: 1.5rem; background-color: #f0f8ff; border-radius: 10px; text-align: center;'>
                        <h2 style='color:#4B9CD3;'>📅 抽せん日: {latest_row['抽せん日']}</h2>
                        <p style='font-size: 2.8rem; color: #FF4B4B; margin: 0.5em 0;'>🎯 <strong>予測:</strong> {latest_row['予測2']}</p>
                        <p style='font-size: 2.4rem; color: #00aa88; margin: 0.5em 0;'>💡 <strong>予測:</strong> {latest_row['予測1']}</p>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ ファイルの読み込み中にエラーが発生しました: {e}")
        else:
            st.warning("⚠️ 予測結果ファイルが見つかりません。まずは予測を実行してください。")

# ========= 予測評価 =========
elif menu == "📊 予測評価":
    st.markdown("## 📊 予測精度の評価")
    with st.container():
        if st.button("🧪 評価を実行"):
            with st.spinner("評価中..."):
                evaluate_and_summarize_predictions()
            st.success("✅ 評価結果が生成されました！")

        if os.path.exists("evaluation_summary.txt"):
            with open("evaluation_summary.txt", encoding="utf-8") as f:
                summary = f.read()
            st.text_area("📄 評価概要", summary, height=400)

        if os.path.exists("evaluation_result.csv"):
            eval_df = pd.read_csv("evaluation_result.csv")
            st.markdown("### 📋 評価結果（詳細）")
            st.dataframe(eval_df, use_container_width=True)

# ========= 分析グラフ =========
elif menu == "📉 予測分析グラフ":
    st.markdown("## 📉 予測の分析グラフ")
    with st.container():
        if os.path.exists("prediction_analysis.png"):
            st.image("prediction_analysis.png", caption="予測分布とパターン分析", use_column_width=True)
        else:
            st.warning("⚠️ 分析グラフがまだ生成されていません。先に予測を実行してください。")

# ========= 予測結果表示 =========
elif menu == "🧾 予測結果表示":
    st.markdown("## 🧾 最新の予測結果（過去10件）")
    with st.container():
        if os.path.exists("Numbers3_predictions.csv"):
            pred_df = pd.read_csv("Numbers3_predictions.csv")
            st.dataframe(pred_df.sort_values("抽せん日", ascending=False).head(10), use_container_width=True)
        else:
            st.warning("⚠️ 予測結果ファイルが見つかりません。まずは予測を実行してください。")
