import pandas as pd

from ml.training import train_and_select


def main():
    print("Training... this can take up to a minute.")
    report = train_and_select()

    print(f"\nLabelled students used: {report['n_rows']}  {report['class_counts']}")
    print("\nModel comparison (5-fold cross-validation, training data):")
    print(pd.DataFrame(report["comparison"]).round(3).to_string(index=False))

    print(f"\nBest model: {report['best_model']}")
    print("New model on the test set:    ", {k: round(v, 3) for k, v in report["new_metrics"].items()})
    if report["old_metrics"]:
        print("Current model on same test set:", {k: round(v, 3) for k, v in report["old_metrics"].items()})
    print("\nDecision:", report["reason"])


if __name__ == "__main__":
    main()