import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

# Read the saved daily weather CSV file
df = pd.read_csv("weather_daily.csv")

# Parse date and extract year and month
if "time" in df.columns:
    df["date"] = pd.to_datetime(df["time"])
else:
    # Try to infer the date column if named differently
    df["date"] = pd.to_datetime(df.iloc[:, 0])

df["year"] = df["date"].dt.year.astype(str)
df["month_num"] = df["date"].dt.month

# Find the max apparent temperature column
temp_col = None
for col in df.columns:
    if "apparent_temperature" in col and "max" in col:
        temp_col = col
        break
if temp_col is None:
    raise ValueError("No max apparent temperature column found in CSV.")


# Group by month and year, calculate the maximum apparent temperature
apparent_temp_monthly = df.groupby(["month_num", "year"])[temp_col].max().unstack()
apparent_temp_monthly = apparent_temp_monthly.sort_index()

plt.figure(figsize=(14, 8))
ax = sns.heatmap(
    apparent_temp_monthly,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    cbar_kws={"label": "Max Apparent Temperature (°C)"},
    linewidths=0.5,
    linecolor="gray",
    annot_kws={"size": 10},
)
plt.title("Max Apparent Temperature per Month per Year")
plt.ylabel("")
plt.xlabel("")
ax.set_yticklabels([calendar.month_abbr[m] for m in apparent_temp_monthly.index], rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
