import requests
import pandas as pd
from datetime import datetime, timedelta

# Define location (Heathrow Airport)
latitude = 51.4700
longitude = -0.4543

# Define time range (last 5 years)
end_date = datetime.today()
start_date = end_date - timedelta(days=10 * 365)

# Open-Meteo API endpoint
url = "https://archive-api.open-meteo.com/v1/archive"

# Parameters
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "hourly": "wind_speed_10m,wind_direction_10m",
    "timezone": "auto",
}

# Fetch data
response = requests.get(url, params=params)

# Print the full URL
print("Full request URL:", response.url)

# Check if the response is zipped
content_encoding = response.headers.get("Content-Encoding", "")
print("Content-Encoding:", content_encoding)
data = response.json()


# Convert to DataFrame
def deg_to_compass(deg):
    import math

    if deg is None or (isinstance(deg, float) and math.isnan(deg)):
        return None
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = int((deg + 22.5) // 45) % 8
    return directions[ix]


if "hourly" in data:
    df = pd.DataFrame(data["hourly"])
    # Add compass direction column (10m)
    df["wind_direction_compass"] = df["wind_direction_10m"].apply(deg_to_compass)
    # Add month column
    df["month"] = pd.to_datetime(df["time"]).dt.to_period("M")
    # List of all compass directions
    all_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    # Group by month and wind direction, count occurrences
    grouped = (
        df.groupby(["month", "wind_direction_compass"]).size().unstack(fill_value=0)
    )
    # Ensure all directions are present as columns
    for d in all_directions:
        if d not in grouped.columns:
            grouped[d] = 0
    grouped = grouped[all_directions]  # order columns
    # Calculate percentages
    grouped_percent = grouped.div(grouped.sum(axis=1), axis=0) * 100
    # Reset index for pretty printing
    grouped_percent = grouped_percent.reset_index()
    # Print the table
    print("\nPercentage of wind direction per month:")
    print(grouped_percent.round(2).to_string(index=False))

    # Calculate E and W percentages directly from the original data so they sum to 100% (monthly)
    ew_counts = (
        df[df["wind_direction_compass"].isin(["E", "W"])]
        .groupby(["month", "wind_direction_compass"])
        .size()
        .unstack(fill_value=0)
    )
    ew_counts = ew_counts.reindex(columns=["E", "W"], fill_value=0)
    ew_total = ew_counts.sum(axis=1)
    ew_percent = ew_counts.div(ew_total, axis=0) * 100
    ew_percent = ew_percent.fillna(0)
    ew_percent = ew_percent.reset_index()

    # Determine predominance for each month
    def predominant_ew(row):
        if row["E"] > row["W"]:
            return "E"
        elif row["W"] > row["E"]:
            return "W"
        else:
            return "Equal"

    ew_percent["predominant"] = ew_percent.apply(predominant_ew, axis=1)
    print(
        "\nPercentage of E and W wind directions per month (with predominance, E+W=100%):"
    )
    print(ew_percent.round(2).to_string(index=False))

    # Calculate E and W percentages by year
    df["year"] = pd.to_datetime(df["time"]).dt.year
    ew_counts_year = (
        df[df["wind_direction_compass"].isin(["E", "W"])]
        .groupby(["year", "wind_direction_compass"])
        .size()
        .unstack(fill_value=0)
    )
    ew_counts_year = ew_counts_year.reindex(columns=["E", "W"], fill_value=0)
    ew_total_year = ew_counts_year.sum(axis=1)
    ew_percent_year = ew_counts_year.div(ew_total_year, axis=0) * 100
    ew_percent_year = ew_percent_year.fillna(0)
    ew_percent_year = ew_percent_year.reset_index()
    ew_percent_year["predominant"] = ew_percent_year.apply(predominant_ew, axis=1)
    print(
        "\nPercentage of E and W wind directions per year (with predominance, E+W=100%):"
    )
    print(ew_percent_year.round(2).to_string(index=False))
else:
    print("Failed to retrieve data or no data available.")
