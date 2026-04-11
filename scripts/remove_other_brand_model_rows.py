from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Remove rows where "brand" or "model" is "other".',
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the cleaned CSV. Defaults to the input filename with '_cleaned' appended.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def remove_other_rows(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"brand", "model"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    brand_is_other = df["brand"].fillna("").astype(str).str.strip().str.lower() == "other"
    model_is_other = df["model"].fillna("").astype(str).str.strip().str.lower() == "other"
    return df.loc[~(brand_is_other | model_is_other)].copy()


def build_default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = (
        build_default_output_path(input_path)
        if args.output is None
        else resolve_path(args.output)
    )

    df = pd.read_csv(input_path)
    cleaned_df = remove_other_rows(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    removed_rows = len(df) - len(cleaned_df)
    print(f"Saved {len(cleaned_df)} rows to {output_path}")
    print(f"Removed {removed_rows} rows where brand or model was 'other'")


if __name__ == "__main__":
    main()
