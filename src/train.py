import argparse, yaml, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow
import mlflow.sklearn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model_out", required=True)
    args = ap.parse_args()

    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))
    df = pd.read_csv(args.data)

    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    num_cols = ["age", "fare", "sibsp", "parch"]
    cat_cols = ["sex", "embarked", "pclass"]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    model = LogisticRegression(C=params["train"]["C"], max_iter=2000)
    clf = Pipeline(steps=[("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"],
        stratify=y
    )

    mlflow.set_experiment("mlops-titanic")

    with mlflow.start_run():
        mlflow.log_params({
            "dataset_name": params["dataset"]["name"],
            "dataset_version": params["dataset"]["version"],
            "model": params["train"]["model"],
            "test_size": params["train"]["test_size"],
            "random_state": params["train"]["random_state"],
            "C": params["train"]["C"],
        })

        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        pred = clf.predict(X_test)

        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)

        mlflow.log_metric("roc_auc", float(auc))
        mlflow.log_metric("accuracy", float(acc))
        mlflow.sklearn.log_model(clf, artifact_path="model")

        joblib.dump(clf, args.model_out)

        print(f"Saved model to: {args.model_out}")
        print(f"Metrics: roc_auc={auc:.4f} | accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
