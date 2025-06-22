# 🎯 Numbers3 予測AI ダッシュボード

Streamlit によるインタラクティブな「数字選択式宝くじ Numbers3」予測・評価ツールです。

## 🚀 起動方法

```bash
pip install -r requirements.txt
streamlit run numbers3_app.py
```

## 📦 構成

- `numbers3_app.py`：Streamlit UIアプリ
- `numbers3_predictor.py`：AI予測エンジン（LSTM/Transformer/PPOなど統合）
- `scrapingnumbers3.py`：過去データ取得（Mizuhoサイト）
- `numbers3.csv`：履歴データ
- `*.pth`：モデル（重い場合は Git LFS または Releases に分離）
- `evaluation_result.csv`：予測と実績の照合結果
- `evaluation_summary.txt`：評価要約レポート

## 📊 使用技術

- PyTorch, LSTM, Transformer, PPO, GAN
- 予測精度評価・ストレート/ボックス分析
- 自己進化型学習・メタ分類フィルタによる多層補正
