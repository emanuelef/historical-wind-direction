"""
Fast Climate Data Explorer - FastAPI + Vanilla JS
Much faster than Streamlit due to:
- Async data fetching
- Client-side rendering
- No full page re-renders
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import httpx
import pandas as pd
from datetime import datetime
from typing import Optional
import asyncio
from functools import lru_cache
import json

app = FastAPI(title="Climate Data Explorer")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cache for API responses (simple in-memory cache)
data_cache = {}
CACHE_DURATION = 3600  # 1 hour in seconds


def get_cache_key(lat: float, lon: float, data_type: str) -> str:
    return f"{data_type}_{lat:.4f}_{lon:.4f}"


def is_cache_valid(cache_entry: dict) -> bool:
    if not cache_entry:
        return False
    cached_time = cache_entry.get("timestamp", 0)
    return (datetime.now().timestamp() - cached_time) < CACHE_DURATION


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.get("/api/wind/{lat}/{lon}")
async def get_wind_data(lat: float, lon: float):
    """Fetch and process wind direction data."""
    cache_key = get_cache_key(lat, lon, "wind")

    if cache_key in data_cache and is_cache_valid(data_cache[cache_key]):
        return data_cache[cache_key]["data"]

    end_date = datetime.today()
    start_year = end_date.year - 10
    start_date = datetime(start_year, 1, 1)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "wind_speed_10m,wind_direction_10m",
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch wind data")

    data = response.json()
    if "hourly" not in data or not data["hourly"].get("time"):
        raise HTTPException(status_code=404, detail="No data available")

    # Process data
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["wind_direction_compass"] = df["wind_direction_10m"].apply(deg_to_compass)
    df["year"] = df["time"].dt.year.astype(str)
    df["month_num"] = df["time"].dt.month
    df["date"] = df["time"].dt.date

    # E/W wind analysis
    ew_df = df[df["wind_direction_compass"].isin(["E", "W"])]
    ew_counts = ew_df.groupby(["month_num", "year", "wind_direction_compass"]).size().unstack(fill_value=0)
    ew_counts = ew_counts.reindex(columns=["E", "W"], fill_value=0)
    ew_total = ew_counts.sum(axis=1)
    ew_percent = ew_counts.div(ew_total, axis=0) * 100
    ew_percent = ew_percent.fillna(0).reset_index()

    # Heatmap data
    heatmap_data = ew_percent.pivot(index="month_num", columns="year", values="W")
    heatmap_data = heatmap_data.sort_index()

    # Calculate stats
    overall_w = ew_percent["W"].mean()
    overall_e = ew_percent["E"].mean()
    monthly_avg = ew_percent.groupby("month_num")["W"].mean()
    max_w_month = int(monthly_avg.idxmax())

    # Daily predominance for streaks
    daily_counts = ew_df.groupby(["date", "wind_direction_compass"]).size().unstack(fill_value=0)
    daily_counts = daily_counts.reindex(columns=["E", "W"], fill_value=0)
    daily_counts["predominant"] = daily_counts.apply(
        lambda row: "E" if row["E"] > row["W"] else "W" if row["W"] > row["E"] else "Equal", axis=1
    )

    # Find streaks
    top_e_runs = find_wind_streaks(daily_counts, "E")
    top_w_runs = find_wind_streaks(daily_counts, "W")

    result = {
        "heatmap": {
            "months": list(heatmap_data.index),
            "years": list(heatmap_data.columns),
            "values": heatmap_data.fillna(0).values.tolist(),
        },
        "monthly_avg": {
            "months": list(range(1, 13)),
            "E": [ew_percent[ew_percent["month_num"] == m]["E"].mean() for m in range(1, 13)],
            "W": [ew_percent[ew_percent["month_num"] == m]["W"].mean() for m in range(1, 13)],
        },
        "yearly_summary": {
            "years": list(ew_percent.groupby("year")["W"].mean().index),
            "W": list(ew_percent.groupby("year")["W"].mean().values),
        },
        "stats": {
            "avg_westerly": round(overall_w, 1),
            "avg_easterly": round(overall_e, 1),
            "predominant": "Westerly" if overall_w > overall_e else "Easterly",
            "max_w_month": max_w_month,
            "max_w_pct": round(monthly_avg.max(), 1),
        },
        "streaks": {
            "easterly": top_e_runs,
            "westerly": top_w_runs,
        },
    }

    data_cache[cache_key] = {"timestamp": datetime.now().timestamp(), "data": result}
    return result


@app.get("/api/rain/{lat}/{lon}")
async def get_rain_data(lat: float, lon: float):
    """Fetch and process rainfall data."""
    cache_key = get_cache_key(lat, lon, "rain")

    if cache_key in data_cache and is_cache_valid(data_cache[cache_key]):
        return data_cache[cache_key]["data"]

    end_date = datetime.today()
    start_year = end_date.year - 10
    start_date = datetime(start_year, 1, 1)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "precipitation",
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch rain data")

    data = response.json()
    if "hourly" not in data or not data["hourly"].get("time"):
        raise HTTPException(status_code=404, detail="No data available")

    # Process data
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year.astype(str)
    df["month_num"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["date"] = df["time"].dt.date
    df["precipitation"] = df["precipitation"].fillna(0)

    # Daily totals
    daily_rain = df.groupby(["date", "year", "month_num", "day"])["precipitation"].sum().reset_index()
    daily_rain = daily_rain.sort_values("date").reset_index(drop=True)

    # Monthly totals
    monthly_rain = daily_rain.groupby(["month_num", "year"])["precipitation"].sum().reset_index()
    heatmap_data = monthly_rain.pivot(index="month_num", columns="year", values="precipitation")
    heatmap_data = heatmap_data.sort_index()

    # Yearly totals
    yearly_total = daily_rain.groupby("year")["precipitation"].sum()

    # Stats
    daily_rain["is_rainy"] = daily_rain["precipitation"] > 0.1
    total_rainfall = daily_rain["precipitation"].sum()
    num_years = len(daily_rain["year"].unique())
    rainy_days = daily_rain["is_rainy"].sum()
    total_days = len(daily_rain)
    max_daily = daily_rain["precipitation"].max()
    max_daily_idx = daily_rain["precipitation"].idxmax()
    max_daily_date = str(daily_rain.loc[max_daily_idx, "date"])

    # Rainy days by year
    rainy_by_year = daily_rain.groupby("year")["is_rainy"].sum()

    # Find dry/wet streaks
    dry_streaks = find_rain_streaks(daily_rain, is_dry=True)
    wet_streaks = find_rain_streaks(daily_rain, is_dry=False)

    result = {
        "heatmap": {
            "months": list(heatmap_data.index),
            "years": list(heatmap_data.columns),
            "values": heatmap_data.fillna(0).values.tolist(),
        },
        "monthly_avg": {
            "months": list(range(1, 13)),
            "values": [daily_rain[daily_rain["month_num"] == m]["precipitation"].mean() for m in range(1, 13)],
        },
        "yearly_summary": {
            "years": list(yearly_total.index),
            "totals": list(yearly_total.values),
            "rainy_days": list(rainy_by_year.values),
        },
        "stats": {
            "avg_yearly": round(total_rainfall / num_years, 0),
            "rainy_pct": round((rainy_days / total_days) * 100, 1),
            "rainy_days": int(rainy_days),
            "total_days": int(total_days),
            "max_daily": round(max_daily, 1),
            "max_daily_date": max_daily_date,
        },
        "streaks": {
            "dry": dry_streaks,
            "wet": wet_streaks,
        },
    }

    data_cache[cache_key] = {"timestamp": datetime.now().timestamp(), "data": result}
    return result


@app.get("/api/temperature/{lat}/{lon}")
async def get_temperature_data(lat: float, lon: float):
    """Fetch and process temperature data."""
    cache_key = get_cache_key(lat, lon, "temp")

    if cache_key in data_cache and is_cache_valid(data_cache[cache_key]):
        return data_cache[cache_key]["data"]

    end_date = datetime.today()
    start_year = end_date.year - 10
    start_date = datetime(start_year, 1, 1)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "apparent_temperature",
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch temperature data")

    data = response.json()
    if "hourly" not in data or not data["hourly"].get("time"):
        raise HTTPException(status_code=404, detail="No data available")

    # Process data
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year.astype(str)
    df["month_num"] = df["time"].dt.month

    # Daily max
    df_daily = df.set_index("time").resample("D").max(numeric_only=True).reset_index()
    df_daily["year"] = df_daily["time"].dt.year.astype(str)
    df_daily["month_num"] = df_daily["time"].dt.month

    # Monthly max
    temp_monthly = df_daily.groupby(["month_num", "year"])["apparent_temperature"].max().unstack()
    temp_monthly = temp_monthly.reindex(index=range(1, 13))

    # Stats
    max_temp = temp_monthly.max().max()
    avg_temp = temp_monthly.mean().mean()
    hottest_month = int(temp_monthly.mean(axis=1).idxmax())

    result = {
        "heatmap": {
            "months": list(temp_monthly.index),
            "years": list(temp_monthly.columns),
            "values": temp_monthly.fillna(0).values.tolist(),
        },
        "stats": {
            "max_temp": round(max_temp, 1) if pd.notna(max_temp) else None,
            "avg_temp": round(avg_temp, 1) if pd.notna(avg_temp) else None,
            "hottest_month": hottest_month,
        },
    }

    data_cache[cache_key] = {"timestamp": datetime.now().timestamp(), "data": result}
    return result


# Helper functions
def deg_to_compass(deg):
    if pd.isna(deg):
        return None
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = int((deg + 22.5) // 45) % 8
    return directions[ix]


def find_wind_streaks(daily_counts, direction):
    streaks = []
    current_start = None
    current_len = 0

    for date, row in daily_counts.iterrows():
        if row["predominant"] == direction:
            if current_start is None:
                current_start = date
                current_len = 1
            else:
                current_len += 1
        else:
            if current_start is not None and current_len > 1:
                streaks.append({
                    "start": str(current_start),
                    "end": str(date),
                    "days": current_len
                })
            current_start = None
            current_len = 0

    if current_start is not None and current_len > 1:
        streaks.append({
            "start": str(current_start),
            "end": str(list(daily_counts.index)[-1]),
            "days": current_len
        })

    return sorted(streaks, key=lambda x: -x["days"])[:5]


def find_rain_streaks(daily_rain, is_dry=True):
    streaks = []
    current_start = None
    current_len = 0
    dates = daily_rain["date"].tolist()
    conditions = (~daily_rain["is_rainy"] if is_dry else daily_rain["is_rainy"]).tolist()

    for i, (date, cond) in enumerate(zip(dates, conditions)):
        if cond:
            if current_start is None:
                current_start = date
                current_len = 1
            else:
                current_len += 1
        else:
            if current_start is not None and current_len > 1:
                streaks.append({
                    "start": str(current_start),
                    "end": str(dates[i-1]),
                    "days": current_len
                })
            current_start = None
            current_len = 0

    if current_start is not None and current_len > 1:
        streaks.append({
            "start": str(current_start),
            "end": str(dates[-1]),
            "days": current_len
        })

    return sorted(streaks, key=lambda x: -x["days"])[:5]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
