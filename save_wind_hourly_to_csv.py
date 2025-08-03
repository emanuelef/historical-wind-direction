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
print("Full request URL:", response.url)
content_encoding = response.headers.get("Content-Encoding", "")
print("Content-Encoding:", content_encoding)
data = response.json()

if "hourly" in data:
    df = pd.DataFrame(data["hourly"])
    # Save to CSV
    df.to_csv("wind_direction_hourly.csv", index=False)
    print("Saved wind direction and speed data to wind_direction_hourly.csv")
else:
    print("Failed to retrieve data or no data available.")
