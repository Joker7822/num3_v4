import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from numbers3_predictor import (
    main_with_improved_predictions,
    evaluate_and_summarize_predictions
)

# ========= ページ設定・UI =========
st.set_page_config(page_title="Numbers3予測AI", layout="wide")
st.markdown("<h1 style='color:#FF4B4B;'>🎯 Numbers3 予測AI</h1>", unsafe_allow_html=True)

menu = st.sidebar.radio("📌 メニュー", [
    "🧠 最新予測表示",
    "🎯 最新予測実行", 
    "📊 予測評価", 
    "📉 予測分析グラフ", 
    "🧾 予測結果表示"
])
if menu == "予測実行":
    st.subheader("📈 最新予測の実行")

    if st.button("予測を開始"):
        with st.spinner("予測を実行中...しばらくお待ちください"):
            main_with_improved_predictions()
        st.success("予測が完了しました！")

el# 最新予測表示
if "最新予測" in menu:
    st.markdown("## 🧠 最新予測結果")
    with st.container():
        if os.path.exists("Numbers3_predictions.csv"):
            try:
                pred_df = pd.read_csv("Numbers3_predictions.csv")
                latest_row = pred_df.sort_values("抽せん日", ascending=False).iloc[0]

                st.success(f"✅ 最新予測が取得されました")

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
