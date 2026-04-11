from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parent
OHE_FEATURES_PATH = BASE_DIR.parent / "models" / "ohe_features.pkl"
MARKET_FEATURES_PATH = BASE_DIR.parent / "models" / "market_features.pkl"

DEFAULT_POPULAR_LIMIT = 10
DEFAULT_BUDGET_THRESHOLD = 0.10
DEFAULT_MAX_RECOMMENDATION_AGE_YEARS = 4
SUPPORTED_BUDGET_SORTS = {
    "avg_price_desc",
    "avg_price_asc",
    "model_year_desc",
    "model_year_asc",
    "most_popular",
    "least_popular",
}
SUPPORTED_TRANSMISSIONS = {"automatic", "manual"}


app = Flask(__name__)


def load_ohe_features() -> pd.DataFrame:
    return pd.read_pickle(OHE_FEATURES_PATH)


def load_market_features() -> pd.DataFrame:
    return pd.read_pickle(MARKET_FEATURES_PATH)


def get_int_query_param(name: str, default: int, minimum: int = 1) -> tuple[int, str | None]:
    raw_value = request.args.get(name, "").strip()
    if not raw_value:
        return default, None
    try:
        value = int(raw_value)
    except ValueError:
        return default, f"{name} must be a valid integer"
    if value < minimum:
        return default, f"{name} must be at least {minimum}"
    return value, None


def get_float_query_param(name: str, default: float, minimum: float = 0.0) -> tuple[float, str | None]:
    raw_value = request.args.get(name, "").strip()
    if not raw_value:
        return default, None
    try:
        value = float(raw_value)
    except ValueError:
        return default, f"{name} must be a valid number"
    if value < minimum:
        return default, f"{name} must be at least {minimum}"
    return value, None


def get_required_string_query_param(name: str) -> tuple[str | None, str | None]:
    value = request.args.get(name, "").strip()
    if not value:
        return None, f"{name} query parameter is required"
    return value, None


def get_required_int_query_param(name: str, minimum: int | None = None) -> tuple[int | None, str | None]:
    value, error = get_required_string_query_param(name)
    if error:
        return None, error
    try:
        parsed = int(value)
    except ValueError:
        return None, f"{name} must be a valid integer"
    if minimum is not None and parsed < minimum:
        return None, f"{name} must be at least {minimum}"
    return parsed, None


def get_required_float_query_param(name: str, minimum: float | None = None) -> tuple[float | None, str | None]:
    value, error = get_required_string_query_param(name)
    if error:
        return None, error
    try:
        parsed = float(value)
    except ValueError:
        return None, f"{name} must be a valid number"
    if minimum is not None and parsed < minimum:
        return None, f"{name} must be at least {minimum}"
    return parsed, None


def get_bool_query_param(name: str, default: bool = False) -> tuple[bool, str | None]:
    raw_value = request.args.get(name, "").strip().lower()
    if not raw_value:
        return default, None
    if raw_value in {"true", "1", "yes"}:
        return True, None
    if raw_value in {"false", "0", "no"}:
        return False, None
    return default, f"{name} must be a valid boolean"


def normalize_transmission_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def has_query_param(name: str) -> bool:
    return request.args.get(name, "").strip() != ""


def clean_brand_filter(features: pd.DataFrame, brand: str | None) -> pd.DataFrame:
    if not brand:
        return features
    return features[features["brand"].str.lower() == brand.strip().lower()]


def get_budget_sorting(sort_by: str | None) -> tuple[list[str], list[bool]]:
    default_columns = ["avgPrice", "model_year", "count", "brand", "model"]
    default_ascending = [True, False, False, True, True]

    if not sort_by:
        return default_columns, default_ascending

    primary_map = {
        "avg_price_desc": (["avgPrice"], [False]),
        "avg_price_asc": (["avgPrice"], [True]),
        "model_year_desc": (["model_year"], [False]),
        "model_year_asc": (["model_year"], [True]),
        "most_popular": (["count"], [False]),
        "least_popular": (["count"], [True]),
    }
    primary_columns, primary_ascending = primary_map[sort_by]
    remaining_columns = [
        column for column in default_columns
        if column not in primary_columns
    ]
    remaining_ascending = [
        ascending
        for column, ascending in zip(default_columns, default_ascending)
        if column not in primary_columns
    ]
    return primary_columns + remaining_columns, primary_ascending + remaining_ascending


def serialize_popular_models(features: pd.DataFrame, limit: int = DEFAULT_POPULAR_LIMIT) -> list[dict[str, object]]:
    grouped = (
        features.groupby(["brand", "model"])
        .agg(count=("model", "size"), avgPrice=("price", "mean"))
        .sort_values(["count", "avgPrice", "brand", "model"], ascending=[False, True, True, True])
        .head(limit)
        .reset_index()
    )
    return [
        {
            "brand": row["brand"],
            "model": row["model"],
            "avgPrice": float(row["avgPrice"]),
            "count": int(row["count"]),
        }
        for _, row in grouped.iterrows()
    ]


@app.get("/car_brands")
def car_brands():
    features = load_ohe_features()
    brands = sorted(features["brand"].dropna().astype(str).unique().tolist())
    return jsonify(brands)


@app.get("/car_models")
def car_model():
    brand = request.args.get("brand", "").strip()
    if not brand:
        return jsonify({"error": "brand query parameter is required"}), 400

    features = load_ohe_features()
    models = (
        clean_brand_filter(features, brand)["model"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    return jsonify(models)


@app.get("/market_insights")
def market_insights():
    vehicle_limit, vehicle_limit_error = get_int_query_param("vehicle_limit", DEFAULT_POPULAR_LIMIT)
    if vehicle_limit_error:
        return jsonify({"error": vehicle_limit_error}), 400

    brands_limit_raw = request.args.get("brands_limit", "").strip()
    if brands_limit_raw:
        try:
            brands_limit = int(brands_limit_raw)
        except ValueError:
            return jsonify({"error": "brands_limit must be a valid integer"}), 400
        if brands_limit < 1:
            return jsonify({"error": "brands_limit must be at least 1"}), 400
    else:
        brands_limit = None

    features = load_market_features()

    average_price_by_brand = (
        features.groupby("brand")
        .agg(avgPrice=("price", "mean"))
        .sort_values(["avgPrice", "brand"], ascending=[True, True])
        .reset_index()
    )
    if brands_limit is not None:
        average_price_by_brand = average_price_by_brand.head(brands_limit)

    payload = {
        "averagePriceByBrand": [
            {"brand": row["brand"], "avgPrice": float(row["avgPrice"])}
            for _, row in average_price_by_brand.iterrows()
        ],
        "mostPopularVehicles": serialize_popular_models(features, limit=vehicle_limit),
    }
    return jsonify(payload)


@app.get("/budget-recommendation")
def budget_recommendation():
    budget_raw = request.args.get("budget", "").strip()
    brand = request.args.get("brand", "").strip()
    sort_by = request.args.get("sort_by", "").strip().lower()

    if not budget_raw:
        return jsonify({"error": "budget query parameter is required"}), 400

    if sort_by and sort_by not in SUPPORTED_BUDGET_SORTS:
        return jsonify(
            {
                "error": (
                    "sort_by must be one of: "
                    + ", ".join(sorted(SUPPORTED_BUDGET_SORTS))
                )
            }
        ), 400

    try:
        budget = float(budget_raw)
    except ValueError:
        return jsonify({"error": "budget must be a valid number"}), 400

    popular_limit, limit_error = get_int_query_param("limit", DEFAULT_POPULAR_LIMIT)
    if limit_error:
        return jsonify({"error": limit_error}), 400

    threshold, threshold_error = get_float_query_param("budget_threshold", DEFAULT_BUDGET_THRESHOLD, minimum=0.0)
    if threshold_error:
        return jsonify({"error": threshold_error}), 400

    max_vehicle_age_years, age_error = get_int_query_param(
        "max_age_years",
        DEFAULT_MAX_RECOMMENDATION_AGE_YEARS,
        minimum=0,
    )
    if age_error:
        return jsonify({"error": age_error}), 400

    min_budget = budget * (1 - threshold)
    max_budget = budget * (1 + threshold)
    current_year = datetime.now().year
    min_model_year = current_year - max_vehicle_age_years

    features = load_market_features()
    if "model_year" not in features.columns:
        return jsonify(
            {
                "error": (
                    "This endpoint requires a newer market_features.pkl artifact "
                    "that includes model_year. Retrain the model first."
                )
            }
        ), 409
    filtered = clean_brand_filter(features, brand)
    grouped = (
        filtered.dropna(subset=["model_year"])
        .groupby(["brand", "model", "model_year"])
        .agg(count=("model", "size"), avgPrice=("price", "mean"))
        .reset_index()
    )
    grouped = grouped[grouped["model_year"] >= min_model_year]
    grouped = grouped[grouped["avgPrice"].between(min_budget, max_budget)]
    sort_columns, sort_ascending = get_budget_sorting(sort_by or None)
    grouped = grouped.sort_values(sort_columns, ascending=sort_ascending).head(popular_limit)

    payload = {
        "budget": budget,
        "brandFilter": brand or None,
        "sortBy": sort_by or None,
        "threshold": threshold,
        "maxVehicleAgeYears": max_vehicle_age_years,
        "recommendations": [
            {
                "brand": row["brand"],
                "model": row["model"],
                "year": int(row["model_year"]),
                "avgPrice": float(row["avgPrice"]),
                "count": int(row["count"]),
            }
            for _, row in grouped.iterrows()
        ],
    }
    return jsonify(payload)


@app.get("/car-comparison")
def car_comparison():
    recommendation_limit, recommendation_limit_error = get_int_query_param("limit", DEFAULT_POPULAR_LIMIT)
    if recommendation_limit_error:
        return jsonify({"error": recommendation_limit_error}), 400

    brand, brand_error = get_required_string_query_param("brand")
    if brand_error:
        return jsonify({"error": brand_error}), 400

    model, model_error = get_required_string_query_param("model")
    if model_error:
        return jsonify({"error": model_error}), 400

    transmission, transmission_error = get_required_string_query_param("transmission")
    if transmission_error:
        return jsonify({"error": transmission_error}), 400
    transmission = transmission.lower()
    if transmission not in SUPPORTED_TRANSMISSIONS:
        return jsonify(
            {"error": "transmission must be either automatic or manual"}
        ), 400

    year, year_error = get_required_int_query_param("year", minimum=1)
    if year_error:
        return jsonify({"error": year_error}), 400

    km, km_error = get_required_int_query_param("km", minimum=0)
    if km_error:
        return jsonify({"error": km_error}), 400

    price, price_error = get_required_float_query_param("price", minimum=0.0)
    if price_error:
        return jsonify({"error": price_error}), 400

    same_brand, same_brand_error = get_bool_query_param("sameBrand", default=False)
    if same_brand_error:
        return jsonify({"error": same_brand_error}), 400

    same_transmission, same_transmission_error = get_bool_query_param("sameTransmission", default=False)
    if same_transmission_error:
        return jsonify({"error": same_transmission_error}), 400

    same_year, same_year_error = get_bool_query_param("sameYear", default=False)
    if same_year_error:
        return jsonify({"error": same_year_error}), 400

    year_from, year_from_error = get_int_query_param("yearFrom", year, minimum=1)
    if year_from_error:
        return jsonify({"error": year_from_error}), 400

    year_to, year_to_error = get_int_query_param("yearTo", year, minimum=1)
    if year_to_error:
        return jsonify({"error": year_to_error}), 400

    km_from, km_from_error = get_int_query_param("kmFrom", km, minimum=0)
    if km_from_error:
        return jsonify({"error": km_from_error}), 400

    km_to, km_to_error = get_int_query_param("kmTo", km, minimum=0)
    if km_to_error:
        return jsonify({"error": km_to_error}), 400

    if year_from > year_to:
        return jsonify({"error": "yearFrom must be less than or equal to yearTo"}), 400

    if km_from > km_to:
        return jsonify({"error": "kmFrom must be less than or equal to kmTo"}), 400

    features = load_market_features()
    required_columns = {"brand", "model", "model_year", "km", "transmission", "price"}
    missing_columns = sorted(required_columns - set(features.columns))
    if missing_columns:
        return jsonify(
            {
                "error": (
                    "This endpoint requires a newer market_features.pkl artifact "
                    "that includes: " + ", ".join(missing_columns)
                )
            }
        ), 409
    features = features.copy()
    features["transmission"] = normalize_transmission_series(features["transmission"])
    has_custom_year_filter = same_year or has_query_param("yearFrom") or has_query_param("yearTo")
    has_custom_km_filter = has_query_param("kmFrom") or has_query_param("kmTo")

    comparison_pool = features.dropna(subset=["brand", "model", "model_year", "km", "price"]).copy()
    comparison_pool = comparison_pool[
        (comparison_pool["brand"].astype(str).str.lower() == brand.lower())
        & (comparison_pool["model"].astype(str).str.lower() == model.lower())
        & (comparison_pool["model_year"] == year)
    ]
    comparison_basis = "brand_model_year_km"
    exact_comparison_pool = comparison_pool[comparison_pool["km"] == km]

    if exact_comparison_pool.empty:
        comparison_basis = "brand_model_year"
        comparison_pool = comparison_pool.copy()
    else:
        comparison_pool = exact_comparison_pool

    if comparison_pool.empty:
        return jsonify(
            {
                "error": (
                    "No vehicles were found for the provided brand, model, year, and km combination or the brand, model, and year fallback"
                )
            }
        ), 404

    lowest_price = float(comparison_pool["price"].min())
    highest_price = float(comparison_pool["price"].max())

    if price < lowest_price:
        selling_price_qualification = "below_range"
    elif price > highest_price:
        selling_price_qualification = "above_range"
    else:
        selling_price_qualification = "within_range"

    recommendations = features.dropna(subset=["brand", "model", "model_year", "km", "price"]).copy()
    if same_year:
        recommendations = recommendations[recommendations["model_year"] == year]
    elif has_custom_year_filter:
        recommendations = recommendations[recommendations["model_year"].between(year_from, year_to)]
    else:
        recommendations = recommendations[recommendations["model_year"] >= year]

    if has_custom_km_filter:
        recommendations = recommendations[recommendations["km"].between(km_from, km_to)]
    else:
        recommendations = recommendations[recommendations["km"] <= km]

    if same_brand:
        recommendations = recommendations[
            recommendations["brand"].astype(str).str.lower() == brand.lower()
        ]
        recommendations = recommendations[
            recommendations["model"].astype(str).str.lower() != model.lower()
        ]
    else:
        recommendations = recommendations[
            recommendations["brand"].astype(str).str.lower() != brand.lower()
        ]

    if same_transmission:
        recommendations = recommendations[
            recommendations["transmission"] == transmission
        ]

    grouped_recommendations = (
        recommendations.groupby(["brand", "model", "model_year", "transmission"])
        .agg(avgPrice=("price", "mean"), avgKm=("km", "mean"), count=("model", "size"))
        .reset_index()
    )
    grouped_recommendations = grouped_recommendations[
        grouped_recommendations["avgPrice"].between(lowest_price, highest_price)
    ]
    grouped_recommendations = (
        grouped_recommendations
        .sort_values(
            ["model_year", "avgKm", "count", "avgPrice", "brand", "model"],
            ascending=[False, True, False, True, True, True],
        )
        .head(recommendation_limit)
    )

    payload = {
        "input": {
            "brand": brand,
            "model": model,
            "year": year,
            "transmission": transmission,
            "km": km,
            "price": price,
            "limit": recommendation_limit,
            "sameBrand": same_brand,
            "sameTransmission": same_transmission,
            "sameYear": same_year,
            "yearFrom": year if same_year else (year_from if has_custom_year_filter else None),
            "yearTo": year if same_year else (year_to if has_custom_year_filter else None),
            "kmFrom": km_from if has_custom_km_filter else None,
            "kmTo": km_to if has_custom_km_filter else None,
        },
        "comparisonBasis": comparison_basis,
        "priceRange": {
            "lowestPrice": lowest_price,
            "highestPrice": highest_price,
        },
        "sellingPriceQualification": selling_price_qualification,
        "vehicleRecommendations": [
            {
                "brand": row["brand"],
                "model": row["model"],
                "year": int(row["model_year"]),
                "transmission": row["transmission"],
                "km": int(round(row["avgKm"])),
                "avgPrice": float(row["avgPrice"]),
            }
            for _, row in grouped_recommendations.iterrows()
        ],
    }
    return jsonify(payload)


@app.get("/most-popular-brands")
def most_popular_brands():
    popular_limit, error = get_int_query_param("limit", DEFAULT_POPULAR_LIMIT)
    if error:
        return jsonify({"error": error}), 400

    features = load_market_features()
    grouped = (
        features.groupby("brand")
        .agg(count=("brand", "size"), avgPrice=("price", "mean"))
        .sort_values(["count", "avgPrice", "brand"], ascending=[False, True, True])
        .head(popular_limit)
        .reset_index()
    )
    return jsonify(
        [
            {
                "brand": row["brand"],
                "avgPrice": float(row["avgPrice"]),
                "count": int(row["count"]),
            }
            for _, row in grouped.iterrows()
        ]
    )


@app.get("/most-popular-models")
def most_popular_model():
    brand = request.args.get("brand", "").strip()
    if not brand:
        return jsonify({"error": "brand query parameter is required"}), 400

    popular_limit, error = get_int_query_param("limit", DEFAULT_POPULAR_LIMIT)
    if error:
        return jsonify({"error": error}), 400

    features = load_market_features()
    filtered = clean_brand_filter(features, brand)
    grouped = (
        filtered.groupby(["brand", "model"])
        .agg(count=("model", "size"), avgPrice=("price", "mean"))
        .sort_values(["count", "avgPrice", "model"], ascending=[False, True, True])
        .head(popular_limit)
        .reset_index()
    )
    return jsonify(
        [
            {
                "brand": row["brand"],
                "model": row["model"],
                "avgPrice": float(row["avgPrice"]),
                "count": int(row["count"]),
            }
            for _, row in grouped.iterrows()
        ]
    )


if __name__ == "__main__":
    app.run()
