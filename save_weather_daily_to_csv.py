import requests
import pandas as pd
from datetime import datetime, timedelta

# Define location (Heathrow Airport)
latitude = 51.4700
longitude = -0.4543

# Define time range (last 10 years)
end_date = datetime.today()
start_date = end_date - timedelta(days=20 * 365)

# Open-Meteo API endpoint
url = "https://archive-api.open-meteo.com/v1/archive"

# Parameters for daily data
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "daily": "temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,relative_humidity_2m_max,relative_humidity_2m_min,cloud_cover_mean,precipitation_sum",
    "timezone": "auto",
}

# Fetch data
response = requests.get(url, params=params)
print("Full request URL:", response.url)
content_encoding = response.headers.get("Content-Encoding", "")
print("Content-Encoding:", content_encoding)
data = response.json()

if "daily" in data:
    df = pd.DataFrame(data["daily"])
    # Save to CSV
    df.to_csv("weather_daily.csv", index=False)
    print("Saved daily weather data to weather_daily.csv")
else:
    print("Failed to retrieve data or no data available.")
