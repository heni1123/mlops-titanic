import argparse, json, yaml, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    df = pd.read_csv(args.data)

    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"],
        stratify=y
    )

    clf = joblib.load(args.model)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)

    out = {
        "dataset_name": params["dataset"]["name"],
        "dataset_version": params["dataset"]["version"],
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
    }

    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved metrics to", args.metrics)
    print(out)

if __name__ == "__main__":
    main()
