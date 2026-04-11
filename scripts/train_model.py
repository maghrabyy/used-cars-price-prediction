from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the used-car price model from a normalized CSV file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data/processed/normalized_hatla2ee.csv",
        help="Normalized CSV produced by prepare_training_data.py",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory where trained artifacts will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for train/test split.",
    )
    return parser.parse_args()


def build_model(input_dim: int) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=70, activation="relu"))
    model.add(Dense(units=35, activation="relu"))
    model.add(Dense(units=1, activation="linear"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def build_market_features(df: pd.DataFrame) -> pd.DataFrame:
    market_df = df.copy()
    market_df["brand"] = market_df["brand"].fillna("").astype(str).str.strip()
    market_df["model"] = market_df["model"].fillna("").astype(str).str.strip()
    market_df["transmission"] = market_df["transmission"].fillna("").astype(str).str.strip().str.title()
    market_df["fuel"] = market_df["fuel"].fillna("unknown").astype(str)
    return market_df


def build_training_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    current_year = datetime.now().year
    working = df.copy()

    working["age"] = current_year + 1 - working["model_year"]
    working = working[(working["price"] < 12_000_000) | working["price"].isna()]
    working = working[~((working["model_year"] == 2024) & (working["km"] > 100))]
    working["fuel"] = working["fuel"].fillna("unknown").astype(str)

    brand_rank = (
        working.groupby("brand", as_index=False)["price"]
        .mean()
        .sort_values("price", ascending=True)
        .reset_index(drop=True)
    )
    brand_rank["rank"] = np.arange(len(brand_rank))
    brand_to_rank = dict(zip(brand_rank["brand"], brand_rank["rank"]))

    base_df = working[
        ["brand", "model", "model_year", "age", "km", "price", "transmission", "fuel"]
    ].copy()
    base_df["transmission"] = base_df["transmission"].fillna("").astype(str).str.strip().str.title()
    base_df["brand_rank"] = base_df["brand"].map(brand_to_rank).astype("float32")

    dummies = pd.get_dummies(base_df[["model", "fuel"]], dtype="float32")

    pd_df = base_df.copy()
    pd_df["transmission"] = np.where(
        pd_df["transmission"].str.lower() == "automatic",
        1,
        0,
    )
    pd_df = pd.concat([pd_df, dummies], axis=1)

    target = np.log1p(pd_df["price"].astype("float32"))
    model_df = pd_df.drop(["brand", "model", "model_year", "price", "fuel"], axis=1)
    model_df = model_df.astype("float32")

    ohe_features_df = pd.concat([base_df, dummies], axis=1)
    ohe_features_df = ohe_features_df.drop(["age", "fuel"], axis=1)
    ohe_features_df = ohe_features_df.filter(regex=r"^(?!fuel)")

    ohe_fuel_df = pd.concat([base_df[["fuel"]], dummies.filter(like="fuel", axis=1)], axis=1)
    ohe_fuel_df = ohe_fuel_df.drop_duplicates(subset="fuel").copy()
    ohe_fuel_df["fuel"] = ohe_fuel_df["fuel"].replace("gas", "benzene")

    return model_df, target, ohe_features_df, ohe_fuel_df


def evaluate_model(
    model_df: pd.DataFrame,
    target: pd.Series,
    epochs: int,
    batch_size: int,
    test_size: float,
    random_state: int,
) -> float:
    x_train, x_test, y_train, y_test = train_test_split(
        model_df,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = RobustScaler()
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train["km"] = scaler.fit_transform(x_train[["km"]]).ravel()
    x_test["km"] = scaler.transform(x_test[["km"]]).ravel()

    model = build_model(x_train.shape[1])
    model.fit(x_train.values, y_train.values, epochs=epochs, batch_size=batch_size, verbose=0)
    predictions = model.predict(x_test.values, verbose=0).ravel()
    return float(r2_score(np.expm1(y_test.values), np.expm1(predictions)))


def train_final_model(
    model_df: pd.DataFrame,
    target: pd.Series,
    epochs: int,
    batch_size: int,
) -> tuple[Sequential, RobustScaler]:
    scaler = RobustScaler()
    final_features = model_df.copy()
    final_features["km"] = scaler.fit_transform(final_features[["km"]]).ravel()

    model = build_model(final_features.shape[1])
    model.fit(final_features.values, target.values, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, scaler


def save_artifacts(
    model: Sequential,
    scaler: RobustScaler,
    ohe_features_df: pd.DataFrame,
    ohe_fuel_df: pd.DataFrame,
    market_features_df: pd.DataFrame,
    models_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "trained_model.h5")
    joblib.dump(scaler, models_dir / "trained_scaler.pkl")
    ohe_features_df.to_pickle(models_dir / "ohe_features.pkl")
    ohe_fuel_df.to_pickle(models_dir / "ohe_fuel.pkl")
    market_features_df.to_pickle(models_dir / "market_features.pkl")


def main() -> None:
    args = parse_args()
    if not args.input.is_absolute():
        args.input = REPO_ROOT / args.input
    if not args.models_dir.is_absolute():
        args.models_dir = REPO_ROOT / args.models_dir
    df = pd.read_csv(args.input)
    market_features_df = build_market_features(df)
    model_df, target, ohe_features_df, ohe_fuel_df = build_training_frames(df)

    r2 = evaluate_model(
        model_df=model_df,
        target=target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model, scaler = train_final_model(
        model_df=model_df,
        target=target,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    save_artifacts(
        model,
        scaler,
        ohe_features_df,
        ohe_fuel_df,
        market_features_df,
        args.models_dir,
    )

    print(f"Training rows: {len(model_df)}")
    print(f"Evaluation R^2: {r2:.4f}")
    print(f"Saved artifacts to {args.models_dir}")


if __name__ == "__main__":
    main()
