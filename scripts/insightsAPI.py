from __future__ import annotations

import calendar
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

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
SUPPORTED_BRAND_SORTS = {"popular", "avgPriceDesc", "avgPriceAsc"}
SUPPORTED_TRANSMISSIONS = {"automatic", "manual"}

GLOBAL_INFLATION_URL = "https://globalinflation.org/api/v1/countries/eg/rates/"
DEFAULT_MARKET_TRENDS_YEARS = 6
DEFAULT_MARKET_TRENDS_MONTHS = 6
GLOBAL_INFLATION_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours
_global_inflation_cache: dict[str, Any] = {"fetched_at": None, "payload": None}


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


def get_int_query_param_from_names(
    names: list[str],
    default: int,
    minimum: int = 1,
) -> tuple[int, str | None]:
    for name in names:
        raw_value = request.args.get(name, "").strip()
        if not raw_value:
            continue
        try:
            value = int(raw_value)
        except ValueError:
            return default, f"{name} must be a valid integer"
        if value < minimum:
            return default, f"{name} must be at least {minimum}"
        return value, None
    return default, None


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


def _cache_is_fresh(fetched_at: datetime | None, ttl_seconds: int) -> bool:
    if fetched_at is None:
        return False
    return (datetime.now() - fetched_at).total_seconds() < ttl_seconds


def fetch_global_inflation_rates(force_refresh: bool = False) -> list[dict[str, Any]]:
    fetched_at = _global_inflation_cache.get("fetched_at")
    if not force_refresh and _cache_is_fresh(fetched_at, GLOBAL_INFLATION_CACHE_TTL_SECONDS):
        payload = _global_inflation_cache.get("payload") or {}
        rates = payload.get("rates") or []
        if isinstance(rates, list):
            return rates

    request_obj = Request(
        GLOBAL_INFLATION_URL,
        headers={"User-Agent": "used-cars-price-prediction/insightsAPI"},
        method="GET",
    )
    with urlopen(request_obj, timeout=10) as response:
        raw = response.read().decode("utf-8")
    payload = json.loads(raw) if raw else {}
    _global_inflation_cache["fetched_at"] = datetime.now()
    _global_inflation_cache["payload"] = payload
    rates = payload.get("rates") or []
    return rates if isinstance(rates, list) else []


def build_preferred_annual_rates(
    rates: list[dict[str, Any]],
    years: list[int] | None = None,
) -> dict[int, float]:
    """
    GlobalInflation may return multiple sources for the same year.
    We keep a single annual rate per year, prioritizing World Bank when present,
    and falling back to any other source when it's not.
    """
    allowed_years = set(years) if years is not None else None
    by_year: dict[int, list[dict[str, Any]]] = {}
    for item in rates:
        if not isinstance(item, dict):
            continue
        try:
            year = int(item.get("year"))
            rate_annual = float(item.get("rate_annual"))
        except (TypeError, ValueError):
            continue
        if allowed_years is not None and year not in allowed_years:
            continue
        normalized = dict(item)
        normalized["_rate_annual"] = rate_annual
        by_year.setdefault(year, []).append(normalized)

    preferred: dict[int, float] = {}
    for year, items in by_year.items():
        if not items:
            continue
        worldbank = [
            it for it in items
            if str(it.get("source", "")).strip().lower() == "worldbank"
        ]
        chosen = worldbank[0] if worldbank else items[0]
        preferred[year] = float(chosen["_rate_annual"])
    return preferred


def annual_to_monthly(rate_annual: float) -> float:
    return (1 + rate_annual / 100) ** (1 / 12) - 1


def get_last_6_months(now: datetime | None = None) -> list[dict[str, int]]:
    now = now or datetime.now()
    result: list[dict[str, int]] = []
    year = now.year
    month = now.month
    for i in range(DEFAULT_MARKET_TRENDS_MONTHS - 1, -1, -1):
        m = month - i
        y = year
        while m <= 0:
            m += 12
            y -= 1
        while m > 12:
            m -= 12
            y += 1
        result.append({"year": y, "month": m})
    return result


def _get_closest_year_value(year: int, annual_rates: dict[int, float]) -> float | None:
    if year in annual_rates:
        return annual_rates[year]
    if not annual_rates:
        return None
    available_years = sorted(annual_rates)
    lower_or_equal = [y for y in available_years if y <= year]
    if lower_or_equal:
        return annual_rates[lower_or_equal[-1]]
    return annual_rates[available_years[0]]


def ease_in_out(t: float) -> float:
    """Smooth easing (0..1 -> 0..1) for nicer trend visualization."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)  # smoothstep


def _visual_wiggle(
    index: int,
    total: int,
    base_rate_percent: float,
    year: int,
    month: int,
) -> float:
    """
    Small deterministic wiggle (in percentage points) to avoid perfectly-flat
    monthly lines when we only have annual rates.
    """
    if total <= 1:
        return 0.0
    t = index / (total - 1)
    phase = (year * 13 + month) % 12
    amplitude = max(0.08, abs(base_rate_percent) * 0.01)  # 0.08pp minimum
    return amplitude * math.sin(2 * math.pi * (t + phase / 12))


def interpolate_monthly_series(
    months: list[dict[str, int]],
    annual_rates: dict[int, float],
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    total = len(months)
    for index, entry in enumerate(months):
        year = int(entry["year"])
        month = int(entry["month"])
        current_annual = _get_closest_year_value(year, annual_rates)
        if current_annual is None:
            continue
        prev_annual = _get_closest_year_value(year - 1, annual_rates) or current_annual

        prev_monthly = annual_to_monthly(prev_annual)
        current_monthly = annual_to_monthly(current_annual)

        t = (month - 1) / 11  # normalize position across 12 months
        interpolated = prev_monthly + (current_monthly - prev_monthly) * ease_in_out(t)
        base_rate_percent = interpolated * 100
        rate_percent = base_rate_percent + _visual_wiggle(index, total, base_rate_percent, year, month)
        series.append(
            {
                "date": f"{year}-{str(month).zfill(2)}",
                "rate": round(rate_percent, 2),
            }
        )
    return series


def build_market_trends() -> dict[str, Any]:
    try:
        rates = fetch_global_inflation_rates()
        current_year = datetime.now().year
        yearly_years = list(range(current_year - (DEFAULT_MARKET_TRENDS_YEARS - 1), current_year + 1))
        # Monthly interpolation may need the previous year when the last 6 months cross year boundaries.
        monthly_years = list(range(current_year - DEFAULT_MARKET_TRENDS_YEARS, current_year + 1))
        annual_rates = build_preferred_annual_rates(rates, years=sorted(set(yearly_years + monthly_years)))
    except Exception:
        return {"monthly": [], "yearly": []}

    if not annual_rates:
        return {"monthly": [], "yearly": []}

    yearly = [{"label": str(year), "value": float(annual_rates[year])} for year in yearly_years if year in annual_rates]

    months = get_last_6_months()
    monthly_series = interpolate_monthly_series(months, annual_rates)
    monthly = []
    for entry, computed in zip(months, monthly_series):
        month_label = calendar.month_abbr[int(entry["month"])] or str(entry["month"])
        monthly.append({"label": month_label, "value": float(computed["rate"])})

    return {"monthly": monthly, "yearly": yearly}


def normalize_make(make: str) -> str:
    make = (make or "").strip()
    if not make:
        return ""
    return make[:1].upper() + make[1:].lower()


def normalize_model(model: str) -> str:
    model = (model or "").strip()
    if not model:
        return ""
    parts = [part for part in model.split(" ") if part]
    return " ".join(part[:1].upper() + part[1:].lower() for part in parts)


def group_prices_by_ad_date_month(features: pd.DataFrame) -> list[dict[str, Any]]:
    if features.empty:
        return []
    working = features.copy()
    working["_ad_dt"] = pd.to_datetime(working.get("ad_date"), errors="coerce")
    working["price"] = pd.to_numeric(working.get("price"), errors="coerce")
    working = working.dropna(subset=["_ad_dt", "price"])
    if working.empty:
        return []
    working["_period"] = working["_ad_dt"].dt.to_period("M").astype(str)  # YYYY-MM
    grouped = (
        working.groupby("_period", as_index=False)["price"]
        .mean()
        .sort_values("_period", ascending=True)
    )
    return [{"period": row["_period"], "avgPrice": float(row["price"])} for _, row in grouped.iterrows()]


def group_prices_by_ad_date_day(features: pd.DataFrame) -> list[dict[str, Any]]:
    if features.empty:
        return []
    working = features.copy()
    working["_ad_dt"] = pd.to_datetime(working.get("ad_date"), errors="coerce")
    working["price"] = pd.to_numeric(working.get("price"), errors="coerce")
    working = working.dropna(subset=["_ad_dt", "price"])
    if working.empty:
        return []
    working["_period"] = working["_ad_dt"].dt.strftime("%Y-%m-%d")
    grouped = (
        working.groupby("_period", as_index=False)["price"]
        .mean()
        .sort_values("_period", ascending=True)
    )
    return [{"period": row["_period"], "avgPrice": float(row["price"])} for _, row in grouped.iterrows()]


def get_trend(data: list[dict[str, Any]]) -> dict[str, Any]:
    if not data:
        return {"priceTrend": "low", "changePercent": 0.0}
    if len(data) == 1:
        return {"priceTrend": "low", "changePercent": 0.0}
    first = float(data[0]["avgPrice"])
    last = float(data[-1]["avgPrice"])
    if first == 0:
        return {"priceTrend": "low", "changePercent": 0.0}
    change_percent = ((last - first) / first) * 100
  

    return {
        "priceTrend": "up" if last > first else "low",
        "changePercent": round(change_percent, 2),
    }


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

    sort_brands = request.args.get("sortBrands", "").strip()
    if sort_brands and sort_brands not in SUPPORTED_BRAND_SORTS:
        return jsonify(
            {
                "error": (
                    "sortBrands must be one of: "
                    + ", ".join(sorted(SUPPORTED_BRAND_SORTS))
                )
            }
        ), 400

    features = load_market_features()

    average_price_by_brand = (
        features.groupby("brand")
        .agg(avgPrice=("price", "mean"), count=("brand", "size"))
        .reset_index()
    )

    if sort_brands == "popular":
        average_price_by_brand = average_price_by_brand.sort_values(
            ["count", "avgPrice", "brand"],
            ascending=[False, True, True],
        )
    elif sort_brands == "avgPriceDesc":
        average_price_by_brand = average_price_by_brand.sort_values(
            ["avgPrice", "brand"],
            ascending=[False, True],
        )
    else:
        average_price_by_brand = average_price_by_brand.sort_values(
            ["avgPrice", "brand"],
            ascending=[True, True],
        )

    if brands_limit is not None:
        average_price_by_brand = average_price_by_brand.head(brands_limit)

    payload = {
        "averagePriceByBrand": [
            {
                "brand": row["brand"],
                "avgPrice": float(row["avgPrice"]),
                "count": int(row["count"]),
            }
            for _, row in average_price_by_brand.iterrows()
        ],
        "mostPopularVehicles": serialize_popular_models(features, limit=vehicle_limit),
        "marketTrends": build_market_trends(),
    }
    return jsonify(payload)


@app.get("/vehicle-price-trend")
def vehicle_price_trend():
    brand, brand_error = get_required_string_query_param("brand")
    if brand_error:
        return jsonify({"error": brand_error}), 400

    model, model_error = get_required_string_query_param("model")
    if model_error:
        return jsonify({"error": model_error}), 400

    year, year_error = get_required_int_query_param("year", minimum=1)
    if year_error:
        return jsonify({"error": year_error}), 400

    make_normalized = normalize_make(brand)
    model_normalized = normalize_model(model)

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
    if "ad_date" not in features.columns:
        return jsonify(
            {
                "error": (
                    "This endpoint requires a newer market_features.pkl artifact "
                    "that includes ad_date. Retrain the model first."
                )
            }
        ), 409

    filtered = features[
        (features["brand"].astype(str).str.strip().str.lower() == make_normalized.lower())
        & (features["model"].astype(str).str.strip().str.lower() == model_normalized.lower())
        & (pd.to_numeric(features["model_year"], errors="coerce") == int(year))
    ]
    if filtered.empty:
        return jsonify({"error": "No market price data found for this vehicle selection"}), 404

    monthly_series = group_prices_by_ad_date_month(filtered)
    group_by = "month"
    series = monthly_series
    if len(monthly_series) <= 1:
        daily_series = group_prices_by_ad_date_day(filtered)
        if daily_series:
            group_by = "day"
            series = daily_series

    if not series:
        return jsonify({"error": "No price history found for this vehicle selection"}), 404

    trend = get_trend(series)

    current_avg_price = float(pd.to_numeric(filtered["price"], errors="coerce").dropna().mean())

    payload = {
        "brand": make_normalized,
        "model": model_normalized,
        "year": int(year),
        "priceTrend": trend["priceTrend"],
        "changePercent": trend["changePercent"],
        "currentAvgPrice": round(current_avg_price, 2),
        "groupBy": group_by,
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
    page_size, page_size_error = get_int_query_param_from_names(
        ["pageSize", "pageSIze"],
        DEFAULT_POPULAR_LIMIT,
    )
    if page_size_error:
        return jsonify({"error": page_size_error}), 400

    page_index, page_index_error = get_int_query_param("pageIndex", 0, minimum=0)
    if page_index_error:
        return jsonify({"error": page_index_error}), 400

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
    grouped_recommendations = grouped_recommendations.sort_values(
        ["model_year", "avgKm", "count", "avgPrice", "brand", "model"],
        ascending=[False, True, False, True, True, True],
    )
    total_recommendations = len(grouped_recommendations)
    start_index = page_index * page_size
    end_index = start_index + page_size
    paged_recommendations = grouped_recommendations.iloc[start_index:end_index]
    has_next = end_index < total_recommendations

    payload = {
        "input": {
            "brand": brand,
            "model": model,
            "year": year,
            "transmission": transmission,
            "km": km,
            "price": price,
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
        "pageInfo": {
            "total": total_recommendations,
            "hasNext": has_next,
            "pageSIze": page_size,
            "pageIndex": page_index,
        },
        "vehicleRecommendations": [
            {
                "brand": row["brand"],
                "model": row["model"],
                "year": int(row["model_year"]),
                "transmission": row["transmission"],
                "km": int(round(row["avgKm"])),
                "avgPrice": float(row["avgPrice"]),
            }
            for _, row in paged_recommendations.iterrows()
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
