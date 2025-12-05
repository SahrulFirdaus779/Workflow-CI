import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run(input_csv: str):
    # Gunakan default local store (.mlruns) saat berjalan di CI
    mlflow.set_experiment("Default")
    df = pd.read_csv(input_csv)
    X = df.drop(columns=df.columns[-1], errors='ignore')
    y = df.iloc[:, -1] if df.shape[1] > 1 else None
    if y is None:
        raise ValueError('Expected target column in last position.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric('accuracy', acc)
        mlflow.sklearn.log_model(model, 'model')
        print(f'Accuracy: {acc:.4f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    run(args.input)
