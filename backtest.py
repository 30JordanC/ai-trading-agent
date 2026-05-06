import numpy as np
import matplotlib.pyplot as plt
from features import FEATURES


def backtest(df, model, retrain_window=100):
    df = df.copy().dropna()

    # =========================
    # CONFIG (VERY IMPORTANT)
    # =========================
    HOLD_PERIOD = 5          # match your 5-day target
    RETRAIN_FREQ = 5        # don't retrain every step
    CONF_THRESHOLD = 0.25
    PROB_THRESHOLD_HIGH = 0.7
    PROB_THRESHOLD_LOW = 0.3

    df["position"] = 0.0

    from model import train_classification_model

    # =========================
    # MAIN LOOP
    # =========================
    for i in range(retrain_window, len(df) - HOLD_PERIOD):

        # ✅ retrain less often (prevents overfitting)
        if i % RETRAIN_FREQ == 0:
            train_slice = df.iloc[i - retrain_window:i]
            model = train_classification_model(train_slice, verbose=False)

        test_row = df.iloc[i:i+1]

        prob = model.predict_proba(test_row[FEATURES])[:, 1][0]
        confidence = abs(prob - 0.5)

        position = 0.0

        # ✅ stronger filter (higher quality trades)
        if prob > PROB_THRESHOLD_HIGH and confidence > CONF_THRESHOLD:
            position = 1.0
        elif prob < PROB_THRESHOLD_LOW and confidence > CONF_THRESHOLD:
            position = -1.0

        # ✅ HOLD POSITION FOR MULTIPLE DAYS (CRITICAL FIX)
        if position != 0:
            df.iloc[i:i+HOLD_PERIOD, df.columns.get_loc("position")] = position

    # =========================
    # NO LOOKAHEAD
    # =========================
    df["position"] = df["position"].shift(1).fillna(0)

    # =========================
    # POSITION CAP
    # =========================
    df["position"] = df["position"].clip(-0.5, 0.5)

    # =========================
    # RETURNS (NOW VALID)
    # =========================
    df["strategy_return"] = df["position"] * df["return"]

    # =========================
    # TRANSACTION COSTS
    # =========================
    df["trade"] = df["position"].diff().abs()
    df["cost"] = df["trade"] * 0.001
    df["strategy_return"] -= df["cost"]

    df["strategy_return"] = df["strategy_return"].fillna(0)

    # =========================
    # BASELINES
    # =========================
    df["buy_hold"] = (1 + df["return"]).cumprod()

    np.random.seed(42)
    random_pos = np.random.choice([-1, 0, 1], size=len(df))
    df["random_return"] = random_pos * df["return"]
    df["random_cum"] = (1 + df["random_return"]).cumprod()

    # =========================
    # FINAL
    # =========================
    df["strategy_cum"] = (1 + df["strategy_return"]).cumprod()

    trades = (df["position"].diff().abs() > 0).sum()
    print(f"\nTotal Trades: {trades}")

    return df


# =========================
# METRICS (UNCHANGED)
# =========================
def sharpe_ratio(returns):
    returns = returns.dropna()
    return 0 if returns.std() == 0 else np.sqrt(252) * returns.mean() / returns.std()


def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()


def print_metrics(df):
    print("\n--- PERFORMANCE ---")

    for name, ret, cum in [
        ("Strategy", df["strategy_return"], df["strategy_cum"]),
        ("Market", df["return"], df["buy_hold"]),
        ("Random", df["random_return"], df["random_cum"]),
    ]:
        sr = sharpe_ratio(ret)
        mdd = max_drawdown(cum)
        win = (ret > 0).mean()

        print(f"\n{name}:")
        print(f"  Sharpe: {sr:.2f}")
        print(f"  Win Rate: {win:.2%}")
        print(f"  Max Drawdown: {mdd:.2%}")


def plot_results(df):
    plt.figure(figsize=(12, 6))

    plt.plot(df["strategy_cum"], label="Strategy")
    plt.plot(df["buy_hold"], label="Buy & Hold")
    plt.plot(df["random_cum"], label="Random")

    plt.legend()
    plt.title("Fixed Strategy (No Leakage) vs Baselines")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")

    plt.show()