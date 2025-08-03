import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

# Read the saved CSV file
csv_path = "wind_direction_hourly.csv"
df = pd.read_csv(csv_path)

# Convert to compass direction if not already present
if "wind_direction_compass" not in df.columns:

    def deg_to_compass(deg):

        if pd.isna(deg):
            return None
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        ix = int((deg + 22.5) // 45) % 8
        return directions[ix]

    df["wind_direction_compass"] = df["wind_direction_10m"].apply(deg_to_compass)

# Add month and year columns
if "month" not in df.columns:
    df["month"] = pd.to_datetime(df["time"]).dt.to_period("M")

# --- Find longest consecutive days with E or W predominance ---
# 1. For each day, find predominant direction (E, W, or Equal)
df["date"] = pd.to_datetime(df["time"]).dt.date
daily_counts = (
    df[df["wind_direction_compass"].isin(["E", "W"])]
    .groupby(["date", "wind_direction_compass"])
    .size()
    .unstack(fill_value=0)
)
daily_counts = daily_counts.reindex(columns=["E", "W"], fill_value=0)


def daily_predominant(row):
    if row["E"] > row["W"]:
        return "E"
    elif row["W"] > row["E"]:
        return "W"
    else:
        return "Equal"


daily_counts["predominant"] = daily_counts.apply(daily_predominant, axis=1)


def find_longest_runs(direction):
    runs = []
    current_start = None
    current_end = None
    current_len = 0
    for date, row in daily_counts.iterrows():
        if row["predominant"] == direction:
            if current_start is None:
                current_start = date
                current_len = 1
            else:
                current_len += 1
            current_end = date
        else:
            if current_start is not None:
                runs.append((current_start, current_end, current_len))
                current_start = None
                current_end = None
                current_len = 0
    # Handle last run
    if current_start is not None:
        runs.append((current_start, current_end, current_len))
    # Sort by length descending, then by start date
    runs = sorted(runs, key=lambda x: (-x[2], x[0]))
    return runs[:10]


# Show only the top 10 periods for E and W in a single table
_top_e = find_longest_runs("E")
_top_w = find_longest_runs("W")
top_periods = pd.DataFrame(
    [("E", start, end, length) for start, end, length in _top_e]
    + [("W", start, end, length) for start, end, length in _top_w],
    columns=["direction", "start_date", "end_date", "days"],
)
top_periods = top_periods.sort_values(["direction", "days"], ascending=[True, False])
print("\nTop 10 longest consecutive days with E or W predominance:")
print(top_periods.to_string(index=False))

# --- Calculate E and W percentages directly from the original data so they sum to 100% (monthly) ---
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

# --- Heatmap of westerly wind percentage per month per year (using ew_percent table directly) ---
# Prepare ew_percent for heatmap: rows=months, columns=years, values=W percentage
ew_percent["year"] = ew_percent["month"].astype(str).str[:4]
ew_percent["month_num"] = ew_percent["month"].astype(str).str[5:7].astype(int)
heatmap_data = ew_percent.pivot(index="month_num", columns="year", values="W")
# Sort by month number
heatmap_data = heatmap_data.sort_index()

plt.figure(figsize=(14, 8))
ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".0f",
    cmap="RdYlGn",
    cbar_kws={"label": "% Westerly (from E/W table)"},
    linewidths=0.5,
    linecolor="gray",
    vmin=0,
    vmax=100,
    annot_kws={"size": 10},
)
plt.title("Westerly Wind Percentage per Month per Year (from E/W table)")
plt.ylabel("")
plt.xlabel("")
ax.set_yticklabels([calendar.month_abbr[m] for m in heatmap_data.index], rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
