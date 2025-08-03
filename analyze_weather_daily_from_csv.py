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

# Use mean cloud cover per month/year
cloud_cover_col = None
for col in df.columns:
    if "cloud" in col and "mean" in col:
        cloud_cover_col = col
        break
if cloud_cover_col is None:
    raise ValueError("No cloud cover mean column found in CSV.")

# --- Count clear days (mean cloud cover < 10%) per year and per month ---
clear_threshold = 10
clear_days = df[df[cloud_cover_col] < clear_threshold]
clear_days_per_year = clear_days.groupby("year").size()
clear_days_per_month = clear_days.groupby(["year", "month_num"]).size()
print("\nNumber of clear days per year (mean cloud cover < 10%):")
print(clear_days_per_year)
print("\nNumber of clear days per month (mean cloud cover < 10%):")
print(clear_days_per_month)

# Group by month and year, calculate mean cloud cover
cloud_cover_monthly = df.groupby(["month_num", "year"])[cloud_cover_col].mean().unstack()
cloud_cover_monthly = cloud_cover_monthly.sort_index()

plt.figure(figsize=(14, 8))
ax = sns.heatmap(
    cloud_cover_monthly,
    annot=True,
    fmt=".0f",
    cmap="Blues",
    cbar_kws={"label": "Mean Cloud Cover (%)"},
    linewidths=0.5,
    linecolor="gray",
    vmin=0,
    vmax=100,
    annot_kws={"size": 10},
)
plt.title("Mean Cloud Cover per Month per Year")
plt.ylabel("")
plt.xlabel("")
ax.set_yticklabels([calendar.month_abbr[m] for m in cloud_cover_monthly.index], rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
