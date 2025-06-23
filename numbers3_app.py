import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from numbers3_predictor import (
    main_with_improved_predictions,
    evaluate_and_summarize_predictions
)

# ページ設定
st.set_page_config(page_title="Numbers3予測AI", layout="wide")
st.markdown("<h1 style='color:#FF4B4B;'>🎯 Numbers3 予測AIダッシュボード</h1>", unsafe_allow_html=True)

# サイドバー
menu = st.sidebar.radio("📌 メニュー", [
    "🧠 最新予測表示", 
    "📊 予測評価", 
    "📉 予測分析グラフ", 
    "🧾 予測結果表示"
])

# 最新予測表示
if "最新予測" in menu:
    st.markdown("## 🧠 最新予測結果")
    with st.container():
        if os.path.exists("Numbers3_predictions.csv"):
            try:
                pred_df = pd.read_csv("Numbers3_predictions.csv")
                latest_row = pred_df.sort_values("抽せん日", ascending=False).iloc[0]

                st.success(f"✅ 最新予測が取得されました（抽せん日: {latest_row['抽せん日']}）")
                st.markdown(f"**🎯 予測2:** `{latest_row['予測2']}`")
                st.markdown(f"**🎯 予測1:** `{latest_row['予測1']}`")
            except Exception as e:
                st.error(f"❌ ファイルの読み込み中にエラーが発生しました: {e}")
        else:
            st.warning("⚠️ 予測結果ファイルが見つかりません。まずは予測を実行してください。")

# 予測評価
elif "予測評価" in menu:
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

# 予測分析グラフ
elif "分析グラフ" in menu:
    st.markdown("## 📉 予測の分析グラフ")
    with st.container():
        if os.path.exists("prediction_analysis.png"):
            st.image("prediction_analysis.png", caption="予測分布とパターン分析", use_column_width=True)
        else:
            st.warning("⚠️ 分析グラフがまだ生成されていません。先に予測を実行してください。")

# 予測結果表示
elif "予測結果" in menu:
    st.markdown("## 🧾 最新の予測結果（過去10件）")
    with st.container():
        if os.path.exists("Numbers3_predictions.csv"):
            pred_df = pd.read_csv("Numbers3_predictions.csv")
            st.dataframe(pred_df.sort_values("抽せん日", ascending=False).head(10), use_container_width=True)
        else:
            st.warning("⚠️ 予測結果ファイルが見つかりません。まずは予測を実行してください。")
