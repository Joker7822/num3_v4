from core.predictor import LotoPredictor
from core.utils import preprocess_data, set_global_seed
import pandas as pd
import os

def main():
    print("[STEP 1] Loading data...")
    set_global_seed(42)
    df = pd.read_csv("data/numbers3.csv")
    X, y, _ = preprocess_data(df)

    predictor = LotoPredictor(input_size=X.shape[1], hidden_size=128)
    predictor.train_model(df)
    preds, _ = predictor.predict(df)
    print(f"[INFO] Predictions: {preds[:3]}")

if __name__ == "__main__":
    main()
