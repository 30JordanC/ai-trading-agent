from data_loader import fetch_stock_data
from features import add_features, add_target
from model import train_classification_model
from backtest import backtest, print_metrics, plot_results


def run_pipeline():
    df = fetch_stock_data("AAPL")

    df = add_features(df)
    df = add_target(df)

    print("\nRunning improved trading strategy...\n")

    results = backtest(df, model=None, retrain_window=100)

    print_metrics(results)
    plot_results(results)


if __name__ == "__main__":
    run_pipeline()