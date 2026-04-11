from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_EXPECTED_COLUMNS = [
    "id",
    "title",
    "year",
    "ad_date",
    "transmission",
    "price",
    "fingerprint",
    "fuel",
    "brand",
    "model",
    "color",
    "class",
    "km",
    "city",
    "detail_link",
]

ENRICHED_MAIN_COLUMNS = [
    "title",
    "company",
    "model",
    "year",
    "price",
    "mileage",
    "color",
    "transmission",
    "location",
    "date_posted",
    "features",
    "detail_link",
    "fuel",
    "body",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw Hatla2ee CSV files into a training-ready CSV.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more raw CSV files created by scrape_hatla2ee.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/processed/normalized_hatla2ee.csv",
        help="Path to the normalized output CSV.",
    )
    return parser.parse_args()


def normalize_transmission(value: str) -> str:
    value = str(value or "").strip().lower()
    if value in {"1", "automatic", "a/t"}:
        return "Automatic"
    return "Manual"


def normalize_fuel(value: str) -> str:
    value = str(value or "").strip().lower()
    mapping = {
        "gas": "benzene",
        "gasoline": "benzene",
        "petrol": "benzene",
        "benzene": "benzene",
        "diesel": "diesel",
        "natural gas": "natural gas",
        "electric": "electric",
        "hybrid": "hybrid",
    }
    return mapping.get(value, value)


def normalize_class(value: str) -> str:
    text = str(value or "").strip()
    replacements = {
        "Automtic": "",
        "Automatic": "",
        "automatic": "",
        "A/T": "",
        "M/T": "",
        "Manual": "",
        "manual": "",
        "F/O": "full option",
        "f/o": "full option",
        "None": "basic",
        " /": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = re.sub(r"\s+", " ", text).strip().lower()
    if text in {"", "none", "basic"}:
        return "standard"
    return text


def extract_numeric(value: str) -> float:
    numbers = re.sub(r"[^0-9.]", "", str(value or ""))
    return float(numbers) if numbers else 0.0


def load_raw_frames(paths: list[Path]) -> pd.DataFrame:
    resolved_paths = [path if path.is_absolute() else REPO_ROOT / path for path in paths]
    frames = [pd.read_csv(path) for path in resolved_paths]
    combined = pd.concat(frames, ignore_index=True)
    if set(RAW_EXPECTED_COLUMNS).issubset(combined.columns):
        return combined[RAW_EXPECTED_COLUMNS].copy()
    if set(ENRICHED_MAIN_COLUMNS).issubset(combined.columns):
        return combined[ENRICHED_MAIN_COLUMNS].copy()
    missing_raw = sorted(set(RAW_EXPECTED_COLUMNS) - set(combined.columns))
    missing_main = sorted(set(ENRICHED_MAIN_COLUMNS) - set(combined.columns))
    raise ValueError(
        "Input CSV did not match a supported schema. "
        f"Missing scraper columns: {missing_raw}. Missing enriched-main columns: {missing_main}."
    )


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        return normalize_main_dataframe(df)

    normalized = pd.DataFrame()
    normalized["car_id"] = df["id"].astype(str).str.strip()
    normalized["title"] = df["title"].fillna("").astype(str).str.strip()
    normalized["model_year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    normalized["ad_date"] = pd.to_datetime(df["ad_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    normalized["transmission"] = df["transmission"].map(normalize_transmission)
    normalized["price"] = df["price"].map(extract_numeric)
    normalized["fingerprint"] = df["fingerprint"].fillna("").astype(str).str.strip()
    normalized["fuel"] = df["fuel"].map(normalize_fuel)
    normalized["brand"] = df["brand"].fillna("").astype(str).str.strip()
    normalized["model"] = df["model"].fillna("").astype(str).str.strip()
    normalized["color"] = df["color"].fillna("").astype(str).str.strip()
    normalized["class"] = df["class"].map(normalize_class)
    normalized["km"] = df["km"].map(extract_numeric)
    normalized["city"] = df["city"].fillna("").astype(str).str.strip()
    normalized["body"] = ""
    normalized["features"] = ""
    normalized["detail_link"] = df["detail_link"].fillna("").astype(str).str.strip()

    normalized = normalized.drop_duplicates(subset="fingerprint", keep="last")
    normalized = normalized[normalized["car_id"] != ""]
    normalized = normalized[normalized["price"] > 0]
    normalized = normalized[normalized["km"] > 0]
    normalized = normalized[normalized["model_year"] > 1900]
    normalized = normalized[normalized["brand"] != ""]
    normalized = normalized[normalized["model"] != ""]
    normalized["ad_date"] = normalized["ad_date"].fillna("")
    return normalized


def normalize_main_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame()
    normalized["car_id"] = (
        df["detail_link"]
        .fillna("")
        .astype(str)
        .str.extract(r"/(\d+)$", expand=False)
        .fillna("")
    )
    normalized["title"] = df["title"].fillna("").astype(str).str.strip()
    normalized["model_year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    normalized["ad_date"] = pd.to_datetime(df["date_posted"], errors="coerce").dt.strftime("%Y-%m-%d")
    normalized["transmission"] = df["transmission"].map(normalize_transmission)
    normalized["price"] = df["price"].map(extract_numeric)
    normalized["fingerprint"] = (
        normalized["car_id"].astype(str)
        + "-"
        + normalized["price"].round().astype(int).astype(str)
    )
    normalized["fuel"] = df["fuel"].map(normalize_fuel).fillna("unknown")
    normalized["brand"] = df["company"].fillna("").astype(str).str.strip()
    normalized["model"] = df["model"].fillna("").astype(str).str.strip()
    normalized["color"] = df["color"].fillna("").astype(str).str.strip()
    normalized["class"] = ""
    normalized["km"] = df["mileage"].map(extract_numeric)
    normalized["city"] = df["location"].fillna("").astype(str).str.strip()
    normalized["body"] = (
        df["body"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    normalized["features"] = df["features"].fillna("").astype(str).str.strip()
    normalized["detail_link"] = df["detail_link"].fillna("").astype(str).str.strip()

    normalized = normalized.drop_duplicates(subset="fingerprint", keep="last")
    normalized = normalized[normalized["car_id"] != ""]
    normalized = normalized[normalized["price"] > 0]
    normalized = normalized[normalized["km"] > 0]
    normalized = normalized[normalized["model_year"] > 1900]
    normalized = normalized[normalized["brand"] != ""]
    normalized = normalized[normalized["model"] != ""]
    normalized["ad_date"] = normalized["ad_date"].fillna("")
    return normalized


def main() -> None:
    args = parse_args()
    if not args.output.is_absolute():
        args.output = REPO_ROOT / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_df = load_raw_frames(args.inputs)
    normalized_df = normalize_dataframe(raw_df)
    normalized_df.to_csv(args.output, index=False)
    print(f"Saved {len(normalized_df)} normalized rows to {args.output}")


if __name__ == "__main__":
    main()
