import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, StackingRegressor,
    GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.linear_model import Ridge, LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
from statsmodels.tsa.arima.model import ARIMA
from stable_baselines3 import PPO
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib
matplotlib.use('Agg')  # ← ★ この行を先に追加！
import matplotlib.pyplot as plt
import aiohttp
from random import shuffle
import asyncio
import warnings
import re
import platform
import gym
import sys
import os
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from neuralforecast.models import TFT
from neuralforecast import NeuralForecast
import onnxruntime
import streamlit as st
from autogluon.tabular import TabularPredictor
import torch.backends.cudnn
from datetime import datetime 
from collections import Counter
import torch.nn.functional as F
import math

# Windows環境のイベントループポリシーを設定
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_global_seed()

import subprocess

def git_commit_and_push(file_path, message):
    try:
        subprocess.run(["git", "add", file_path], check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode != 0:
            subprocess.run(["git", "config", "--global", "user.name", "github-actions"], check=True)
            subprocess.run(["git", "config", "--global", "user.email", "github-actions@github.com"], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            subprocess.run(["git", "push"], check=True)
        else:
            print(f"[INFO] No changes in {file_path}")
    except Exception as e:
        print(f"[WARNING] Git commit/push failed: {e}")

def calculate_reward(selected_numbers, winning_numbers, cycle_scores):
    match_count = len(set(selected_numbers) & set(winning_numbers))
    avg_cycle_score = np.mean([cycle_scores.get(n, 999) for n in selected_numbers])
    reward = match_count * 0.5 + max(0, 1 - avg_cycle_score / 50)
    return reward

class LotoEnv(gym.Env):
    def __init__(self, historical_numbers):
        super(LotoEnv, self).__init__()
        self.historical_numbers = historical_numbers
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self):
        return np.zeros(10, dtype=np.float32)

def step(self, action):
    if action.size == 0:
        return np.zeros(10, dtype=np.float32), -1.0, True, {}

    selected_numbers = set(np.argsort(action)[-3:])
    target_numbers = set(self.target_numbers_list[self.current_index])

    match_count = len(selected_numbers & target_numbers)
    # cycle_scores を self.cycle_scores で持っていない場合は、適当なデフォルト値を使用する
    avg_cycle_score = 999  # 仮に固定値を設定
    reward = match_count * 0.5 + max(0, 1 - avg_cycle_score / 50)

    done = True
    obs = np.zeros(10, dtype=np.float32)
    return obs, reward, done, {}

class DiversityEnv(gym.Env):
    def __init__(self, historical_numbers):
        super(DiversityEnv, self).__init__()
        self.historical_numbers = historical_numbers
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.previous_outputs = set()

    def reset(self):
        return np.zeros(10, dtype=np.float32)

    def step(self, action):
        if action.size == 0:
            return np.zeros(10, dtype=np.float32), -1.0, True, {}  # エラー回避

        selected = np.argsort(action)[-3:]  # または[-4:]

        selected = tuple(sorted(np.argsort(action)[-4:]))
        reward = 1.0 if selected not in self.previous_outputs else -1.0
        self.previous_outputs.add(selected)
        return np.zeros(10, dtype=np.float32), reward, True, {}

class CycleEnv(gym.Env):
    def __init__(self, historical_numbers):
        super(CycleEnv, self).__init__()
        self.historical_numbers = historical_numbers
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.cycle_scores = calculate_number_cycle_score(historical_numbers)

    def reset(self):
        return np.zeros(10, dtype=np.float32)

    def step(self, action):
        if action.size == 0:
            return np.zeros(10, dtype=np.float32), -1.0, True, {}  # エラー回避

        selected = np.argsort(action)[-3:]  # または[-4:]

        selected = np.argsort(action)[-4:]
        avg_cycle = np.mean([self.cycle_scores.get(n, 999) for n in selected])
        reward = max(0, 1 - (avg_cycle / 50))
        return np.zeros(10, dtype=np.float32), reward, True, {}

class ProfitLotoEnv(gym.Env):
    def __init__(self, historical_numbers):
        super(ProfitLotoEnv, self).__init__()
        self.historical_numbers = historical_numbers
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self):
        return np.zeros(10, dtype=np.float32)

    def step(self, action):
        if action.size == 0:
            return np.zeros(10, dtype=np.float32), -1.0, True, {}  # エラー回避

        selected = np.argsort(action)[-3:]  # または[-4:]

        selected = list(np.argsort(action)[-3:])
        reward_table = {
            "ストレート": 90000,
            "ボックス": 10000,
            "ミニ": 4000,
            "はずれ": -200
        }
        best_reward = -200
        for winning in self.historical_numbers:
            result = classify_numbers3_prize(selected, winning)
            reward = reward_table.get(result, -200)
            if reward > best_reward:
                best_reward = reward
        return np.zeros(10, dtype=np.float32), best_reward, True, {}

class MultiAgentPPOTrainer:
    def __init__(self, historical_data, total_timesteps=5000):
        self.historical_data = historical_data
        self.total_timesteps = total_timesteps
        self.agents = {}

    def train_agents(self):
        envs = {
            "accuracy": LotoEnv(self.historical_data),
            "diversity": DiversityEnv(self.historical_data),
            "cycle": CycleEnv(self.historical_data),
            "profit": ProfitLotoEnv(self.historical_data)  # ★ ここを追加
        }

        for name, env in envs.items():
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=self.total_timesteps)
            self.agents[name] = model
            print(f"[INFO] PPO {name} エージェント学習完了")

    def predict_all(self, num_candidates=50):
        predictions = []
        for name, model in self.agents.items():
            obs = model.env.reset()
            for _ in range(num_candidates // 3):
                action, _ = model.predict(obs)
                selected = list(np.argsort(action)[-4:])
                predictions.append((selected, 0.9))  # 信頼度は仮
        return predictions

class AdversarialLotoEnv(gym.Env):
    def __init__(self, target_numbers_list):
        """
        GANが生成した番号（target_numbers_list）をターゲットとし、
        PPOに「それらを当てさせる」対戦環境
        """
        super(AdversarialLotoEnv, self).__init__()
        self.target_numbers_list = target_numbers_list
        self.current_index = 0
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self):
        self.current_index = (self.current_index + 1) % len(self.target_numbers_list)
        return np.zeros(10, dtype=np.float32)

    def step(self, action):
        if action.size == 0:
            return np.zeros(10, dtype=np.float32), -1.0, True, {}

        selected_numbers = set(np.argsort(action)[-3:])
        target_numbers = set(self.target_numbers_list[self.current_index])

        match_count = len(selected_numbers & target_numbers)
        avg_cycle_score = np.mean([self.cycle_scores.get(n, 999) for n in selected_numbers])
        reward = match_count * 0.5 + max(0, 1 - avg_cycle_score / 50)

        done = True
        obs = np.zeros(10, dtype=np.float32)
        return obs, reward, done, {}

def score_real_structure_similarity(numbers):
    """
    数字リストに対して、「本物らしい構造かどうか」を評価するスコア（0〜1）
    - 合計が10〜20
    - 重複がない
    - 並びが昇順 or 降順
    """
    if len(numbers) != 3:
        return 0
    score = 0
    if 10 <= sum(numbers) <= 20:
        score += 1
    if len(set(numbers)) == 3:
        score += 1
    if numbers == sorted(numbers) or numbers == sorted(numbers, reverse=True):
        score += 1
    return score / 3  # 最大3点満点を0〜1スケール

class LotoGAN(nn.Module):
    def __init__(self, noise_dim=100):
        super(LotoGAN, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Sigmoid()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.noise_dim = noise_dim

    def generate_samples(self, num_samples):
        noise = torch.randn(num_samples, self.noise_dim)
        with torch.no_grad():
            samples = self.generator(noise)
        return samples.numpy()

    def evaluate_generated_numbers(self, sample_tensor):
        """
        sample_tensor: shape=(10,) のTensor（0〜1値で各数字のスコア）
        上位3つを選んで番号に変換 → 判別器スコアと構造スコアを合成
        """
        numbers = list(np.argsort(sample_tensor.cpu().numpy())[-3:])
        numbers.sort()
        real_score = score_real_structure_similarity(numbers)

        with torch.no_grad():
            discriminator_score = self.discriminator(sample_tensor.unsqueeze(0)).item()

        final_score = 0.5 * discriminator_score + 0.5 * real_score
        return final_score

class DiffusionNumberGenerator(nn.Module):
    def __init__(self, noise_dim=16, steps=100):
        super(DiffusionNumberGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.steps = steps

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 各数字のスコア（0〜9）
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def generate(self, num_samples=20):
        samples = []
        for _ in range(num_samples):
            noise = torch.randn(1, self.noise_dim)
            x = noise
            for _ in range(self.steps):
                noise_grad = torch.randn_like(x) * 0.1
                x = x - 0.01 * x + noise_grad
            with torch.no_grad():
                scores = self.forward(x).squeeze().numpy()
            top4 = np.argsort(scores)[-4:]
            samples.append(sorted(top4.tolist()))
        return samples

def create_advanced_features(dataframe):
    dataframe = dataframe.copy()
    def convert_to_number_list(x):
        if isinstance(x, str):
            cleaned = x.strip("[]").replace(",", " ").replace("'", "").replace('"', "")
            return [int(n) for n in cleaned.split() if n.isdigit()]
        return x if isinstance(x, list) else [0]

    dataframe['本数字'] = dataframe['本数字'].apply(convert_to_number_list)
    dataframe['抽せん日'] = pd.to_datetime(dataframe['抽せん日'])

    valid_mask = (dataframe['本数字'].apply(len) == 3)
    dataframe = dataframe[valid_mask].copy()

    if dataframe.empty:
        print("[ERROR] 有効な本数字が存在しません（4桁データがない）")
        return pd.DataFrame()  # 空のDataFrameを返す

    nums_array = np.vstack(dataframe['本数字'].values)
    features = pd.DataFrame(index=dataframe.index)

    features['数字合計'] = nums_array.sum(axis=1)
    features['数字平均'] = nums_array.mean(axis=1)
    features['最大'] = nums_array.max(axis=1)
    features['最小'] = nums_array.min(axis=1)
    features['標準偏差'] = np.std(nums_array, axis=1)

    return pd.concat([dataframe, features], axis=1)

def preprocess_data(data):
    """データの前処理: 特徴量の作成 & スケーリング"""
    
    # 特徴量作成
    processed_data = create_advanced_features(data)

    if processed_data.empty:
        print("エラー: 特徴量生成後のデータが空です。データのフォーマットを確認してください。")
        return None, None, None

    print("=== 特徴量作成後のデータ ===")
    print(processed_data.head())

    # 数値特徴量の選択
    numeric_features = processed_data.select_dtypes(include=[np.number]).columns
    X = processed_data[numeric_features].fillna(0)  # 欠損値を0で埋める

    print(f"数値特徴量の数: {len(numeric_features)}, サンプル数: {X.shape[0]}")

    if X.empty:
        print("エラー: 数値特徴量が作成されず、データが空になっています。")
        return None, None, None

    # スケーリング
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print("=== スケーリング後のデータ ===")
    print(X_scaled[:5])  # 最初の5件を表示

    # 目標変数の準備
    try:
        y = np.array([list(map(int, nums)) for nums in processed_data['本数字']])
    except Exception as e:
        print(f"エラー: 目標変数の作成時に問題が発生しました: {e}")
        return None, None, None

    return X_scaled, y, scaler

def convert_numbers_to_binary_vectors(data):
    vectors = []
    for numbers in data['本数字']:
        vec = np.zeros(10)
        for n in numbers:
            if 0 <= n <= 9:
                vec[n] = 1
        vectors.append(vec)
    return np.array(vectors)

def calculate_prediction_errors(predictions, actual_numbers):
    """予測値と実際の当選結果の誤差を計算し、特徴量として保存"""
    errors = []
    for pred, actual in zip(predictions, actual_numbers):
        pred_numbers = set(pred[0])
        actual_numbers = set(actual)
        error_count = len(actual_numbers - pred_numbers)
        errors.append(error_count)
    
    return np.mean(errors)

