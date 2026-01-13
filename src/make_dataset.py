import argparse
import pandas as pd
from sklearn.datasets import fetch_openml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True, choices=["v1", "v2", "v3"])
    args = ap.parse_args()

    # Load Titanic dataset from OpenML
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    df = X.copy()
    df["target"] = y.astype(int)

    # Keep a subset of useful columns
    keep = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "target"]
    df = df[keep]

    # Create dataset versions to satisfy "change dataset > 2 times"
    # v1: keep raw with missing values (will be handled by imputers later)
    # v2: drop rows with missing age
    # v3: drop rows with missing age + embarked
    if args.version == "v2":
        df = df.dropna(subset=["age"])
    elif args.version == "v3":
        df = df.dropna(subset=["age", "embarked"])

    df.to_csv(args.out, index=False)
    print(f"Saved Titanic dataset {args.version} to {args.out} | shape={df.shape}")

if __name__ == "__main__":
    main()
