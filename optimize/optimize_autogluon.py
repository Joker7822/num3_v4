import optuna
from autogluon.tabular import TabularPredictor
import pandas as pd

def optimize_autogluon(X, y, model_index):
    def objective(trial):
        df = pd.DataFrame(X)
        df['target'] = y[:, model_index]
        predictor = TabularPredictor(label='target', verbosity=0).fit(
            df, presets="best_quality", time_limit=60
        )
        return predictor.leaderboard(silent=True).iloc[0]['score_val']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    return study.best_params
