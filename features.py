import pandas as pd
import numpy as np

FEATURES = [
    "return_lag1",
    "return_lag2",
    "volatility",
    "momentum",
    "rsi",
    "macd",
    "volume_change",
    "trend_strength",
    "regime",

    # 🔥 NEW (REAL SIGNAL)
    "z_score",
    "volatility_breakout",
    "trend_persistence"
]


def add_features(df):
    df = df.copy()

    df["return"] = df["close"].pct_change()

    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()

    df["trend_strength"] = (df["ma_5"] - df["ma_10"]) / df["close"]

    df["volatility"] = df["return"].rolling(10).std()

    df["momentum"] = df["close"] / df["close"].shift(5) - 1

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26

    df["volume_change"] = df["volume"].pct_change()

    # Regime
    df["trend"] = df["close"].rolling(50).mean()
    df["regime"] = (df["close"] > df["trend"]).astype(int)

    # =========================
    # 🔥 STRONG FEATURES
    # =========================

    # Mean reversion (z-score)
    rolling_mean = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std()
    df["z_score"] = (df["close"] - rolling_mean) / rolling_std

    # Volatility breakout
    df["volatility_breakout"] = df["close"] / df["close"].rolling(20).max()

    # Trend persistence
    df["trend_persistence"] = df["return"].rolling(5).sum()

    return df


def add_target(df):
    df = df.copy()

    # 🔥 FIX: 5-day prediction (NOT 1-day noise)
    df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)

    return df.dropna()