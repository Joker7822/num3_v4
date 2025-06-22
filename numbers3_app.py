import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from test import (
    main_with_improved_predictions,
    evaluate_and_summarize_predictions
)

st.set_page_config(page_title="Numbers3予測AI", layout="wide")

st.title("🎯 Numbers3 予測AIダッシュボード")

menu = st.sidebar.radio("メニュー", ["予測実行", "予測評価", "予測分析グラフ", "予測結果表示"])

if menu == "予測実行":
    st.subheader("📈 最新予測の実行")

    if st.button("予測を開始"):
        with st.spinner("予測を実行中...しばらくお待ちください"):
            main_with_improved_predictions()
        st.success("予測が完了しました！")

elif menu == "予測評価":
    st.subheader("📊 予測精度の評価")

    if st.button("評価を実行"):
        with st.spinner("評価中..."):
            evaluate_and_summarize_predictions()
        st.success("評価結果が生成されました！")

    if os.path.exists("evaluation_summary.txt"):
        with open("evaluation_summary.txt", encoding="utf-8") as f:
            summary = f.read()
        st.text_area("📄 評価概要", summary, height=500)

    if os.path.exists("evaluation_result.csv"):
        eval_df = pd.read_csv("evaluation_result.csv")
        st.dataframe(eval_df)

elif menu == "予測分析グラフ":
    st.subheader("📉 予測の分析グラフ")

    if os.path.exists("prediction_analysis.png"):
        st.image("prediction_analysis.png", caption="予測分布とパターン分析", use_column_width=True)
    else:
        st.warning("分析グラフがまだ生成されていません。先に予測を実行してください。")

elif menu == "予測結果表示":
    st.subheader("🧾 最新の予測結果")

    if os.path.exists("numbers3_predictions.csv"):
        pred_df = pd.read_csv("numbers3_predictions.csv")
        st.dataframe(pred_df.sort_values("抽せん日", ascending=False).head(10))
    else:
        st.warning("予測結果ファイルが見つかりません。まずは予測を実行してください。")
