from optimize.optimize_autogluon import optimize_autogluon
from core.utils import preprocess_data
import pandas as pd

print("[STEP] Running AutoGluon optimization")
data = pd.read_csv("data/numbers3.csv")
X, y, _ = preprocess_data(data)
best_params = optimize_autogluon(X, y, 0)
print("Best params:", best_params)
