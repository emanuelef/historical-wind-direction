import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# For interactive map
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="Historical Wind Data Explorer", layout="wide")

# Default location: Milan, Italy
latitude = 45.4642
longitude = 9.19


st.markdown("### Select two locations to compare apparent temperature")

# --- State Initialization ---
if "locations" not in st.session_state:
    st.session_state.locations = []
if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = False
if "last_clicked_coords" not in st.session_state:
    st.session_state.last_clicked_coords = None
# Cache for temperature data
if "temp_data_cache" not in st.session_state:
    st.session_state.temp_data_cache = {}

# --- Map Display and Interaction ---
map_center = [latitude, longitude]
m = folium.Map(location=map_center, zoom_start=4, tiles="CartoDB Positron")

# Add markers for existing locations
for i, (lat, lon) in enumerate(st.session_state.locations):
    color = "blue" if i == 0 else "red"
    folium.Marker(
        location=[lat, lon],
        popup=f"Location {i+1}: {lat:.4f}, {lon:.4f}",
        icon=folium.Icon(color=color, icon="info-sign"),
    ).add_to(m)

m.add_child(folium.LatLngPopup())

# Render map and get output
map_output = st_folium(m, width="100%", height=500, key="map")

# Check if map has TRULY changed in a meaningful way (new clicks, not just zoom/pan)
# Use center and bounds as more reliable indicators for real interactions vs zoom/pan
if map_output and map_output.get("last_clicked"):
    current_click = map_output.get("last_clicked")

    # Only process click if it's new AND different from previous click
    if current_click != st.session_state.last_clicked_coords:
        lat, lon = current_click["lat"], current_click["lng"]

        # Save current clicked coordinates to compare next time
        st.session_state.last_clicked_coords = current_click

        # Check if the click is on a new location
        is_new = not any(
            abs(lat - loc[0]) < 1e-4 and abs(lon - loc[1]) < 1e-4
            for loc in st.session_state.locations
        )

        if is_new:
            # This is a true new location selection
            st.session_state.locations.append((lat, lon))
            if len(st.session_state.locations) > 2:
                st.session_state.locations.pop(0)
            # Clear affected cache entry to force refresh
            loc_key = f"{lat:.4f}_{lon:.4f}"
            if loc_key in st.session_state.temp_data_cache:
                del st.session_state.temp_data_cache[loc_key]
            # New location invalidates previous comparison
            st.session_state.compare_mode = False
            st.rerun()

# Remove vertical gap between map and controls
st.markdown(
    "<style>div[data-testid='stVerticalBlock']:has(div.folium-map) + div[data-testid='stVerticalBlock'] { margin-top: -32px !important; } </style>",
    unsafe_allow_html=True,
)

# Coordinates and buttons in a single row below the map
coord_col, btn_col1, btn_col2 = st.columns([2, 1, 2])
with coord_col:
    # Show selected locations
    for i, (lat, lon) in enumerate(st.session_state.locations):
        st.markdown(f"**Location {i+1}:** lat={lat:.4f}, lon={lon:.4f}")
with btn_col1:
    if st.button("Clear Locations"):
        st.session_state.locations = []
        st.session_state.compare_mode = False
        # Clear data cache when locations are cleared
        st.session_state.temp_data_cache = {}
        st.rerun()
with btn_col2:
    if st.button(
        "Compare Apparent Temperature",
        disabled=len(st.session_state.locations) != 2,
    ):
        st.session_state.compare_mode = True
        st.rerun()


def fetch_and_process(lat, lon):
    end_date = datetime.today()
    start_year = end_date.year - 10
    start_date = datetime(start_year, 1, 1)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "wind_speed_10m,wind_direction_10m,apparent_temperature",
        "timezone": "auto",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200 and "hourly" in response.json():
        data = response.json()["hourly"]
        if not data or not data.get("time") or len(data["time"]) == 0:
            return None, "No data available for this location and time range."
        df = pd.DataFrame(data)
        if df.empty:
            return None, "No data returned for this location."
        df["time"] = pd.to_datetime(df["time"])
        if (
            "apparent_temperature" not in df.columns
            or df["apparent_temperature"].isna().all()
        ):
            return None, "No apparent temperature data available for this location."
        df["year"] = df["time"].dt.year.astype(str)
        df["month_num"] = df["time"].dt.month
        df_daily = (
            df.set_index("time").resample("D").max(numeric_only=True).reset_index()
        )
        df_daily["year"] = df_daily["time"].dt.year.astype(str)
        df_daily["month_num"] = df_daily["time"].dt.month
        temp_monthly = (
            df_daily.groupby(["month_num", "year"])["apparent_temperature"]
            .max()
            .unstack()
        )
        if len(temp_monthly.columns) > 0:
            min_year = int(temp_monthly.columns.min())
            max_year = int(temp_monthly.columns.max())
            all_years = [str(y) for y in range(min_year, max_year + 1)]
        else:
            all_years = []
        all_months = pd.Index(range(1, 13), name="month_num")
        temp_monthly = temp_monthly.reindex(index=all_months, columns=all_years)
        temp_monthly = temp_monthly.sort_index()
        return temp_monthly, None
    else:
        return None, "Failed to fetch wind data for this location."


# Only show comparison if compare_mode is True and there are exactly 2 locations
if st.session_state.compare_mode and len(st.session_state.locations) == 2:
    # Get locations
    (lat1, lon1), (lat2, lon2) = st.session_state.locations

    # Create location keys for cache
    loc1_key = f"{lat1:.4f}_{lon1:.4f}"
    loc2_key = f"{lat2:.4f}_{lon2:.4f}"

    # Check if we need to fetch data (only if not cached)
    need_fetch = False
    if (
        loc1_key not in st.session_state.temp_data_cache
        or loc2_key not in st.session_state.temp_data_cache
    ):
        need_fetch = True

    # Fetch only if needed
    if need_fetch:
        with st.spinner("Fetching and processing data for both locations..."):
            if loc1_key not in st.session_state.temp_data_cache:
                temp1, err1 = fetch_and_process(lat1, lon1)
                st.session_state.temp_data_cache[loc1_key] = (temp1, err1)
            else:
                temp1, err1 = st.session_state.temp_data_cache[loc1_key]

            if loc2_key not in st.session_state.temp_data_cache:
                temp2, err2 = fetch_and_process(lat2, lon2)
                st.session_state.temp_data_cache[loc2_key] = (temp2, err2)
            else:
                temp2, err2 = st.session_state.temp_data_cache[loc2_key]
    else:
        # Use cached data
        temp1, err1 = st.session_state.temp_data_cache[loc1_key]
        temp2, err2 = st.session_state.temp_data_cache[loc2_key]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Location 1: lat={lat1:.4f}, lon={lon1:.4f}")
        if err1:
            st.info(err1)
        else:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import calendar
            from matplotlib.colors import LinearSegmentedColormap

            colors = [
                (0.0, "#ADD8E6"),
                (0.4, "#FFFF66"),
                (0.7, "#FFA500"),
                (1.0, "#FF4500"),
            ]
            cmap = LinearSegmentedColormap.from_list("custom_heat", colors, N=256)
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                temp1,
                annot=True,
                fmt=".1f",
                cmap=cmap,
                cbar_kws={"label": "Max Apparent Temperature (°C)"},
                linewidths=0.5,
                linecolor="gray",
                annot_kws={"size": 10},
                vmin=0,
                vmax=45,
            )
            ax3.set_title("Max Apparent Temperature per Month per Year")
            ax3.set_ylabel("")
            ax3.set_xlabel("")
            ax3.set_yticklabels(
                [calendar.month_abbr[m] for m in temp1.index],
                rotation=0,
            )
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig3)
    with col2:
        st.markdown(f"### Location 2: lat={lat2:.4f}, lon={lon2:.4f}")
        if err2:
            st.info(err2)
        else:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import calendar
            from matplotlib.colors import LinearSegmentedColormap

            colors = [
                (0.0, "#ADD8E6"),
                (0.4, "#FFFF66"),
                (0.7, "#FFA500"),
                (1.0, "#FF4500"),
            ]
            cmap = LinearSegmentedColormap.from_list("custom_heat", colors, N=256)
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                temp2,
                annot=True,
                fmt=".1f",
                cmap=cmap,
                cbar_kws={"label": "Max Apparent Temperature (°C)"},
                linewidths=0.5,
                linecolor="gray",
                annot_kws={"size": 10},
                vmin=0,
                vmax=45,
            )
            ax3.set_title("Max Apparent Temperature per Month per Year")
            ax3.set_ylabel("")
            ax3.set_xlabel("")
            ax3.set_yticklabels(
                [calendar.month_abbr[m] for m in temp2.index],
                rotation=0,
            )
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig3)

    # Difference heatmap (Location 2 - Location 1)
    if (err1 is None) and (err2 is None):
        st.markdown("### Difference: Location 2 minus Location 1 (°C)")
        # Align both DataFrames to the same index/columns
        temp1_aligned, temp2_aligned = temp1.align(temp2, join="outer")
        diff = temp2_aligned - temp1_aligned
        # Use a diverging colormap centered at 0
        import matplotlib.pyplot as plt
        import seaborn as sns
        import calendar

        fig_diff, ax_diff = plt.subplots(figsize=(14, 4))
        sns.heatmap(
            diff,
            annot=True,
            fmt=".1f",
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": "Δ Max Apparent Temperature (°C) (Loc2 - Loc1)"},
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 10},
            vmin=-20,
            vmax=20,
        )
        ax_diff.set_title(
            "Difference in Max Apparent Temperature per Month per Year (Loc2 - Loc1)"
        )
        ax_diff.set_ylabel("")
        ax_diff.set_xlabel("")
        ax_diff.set_yticklabels(
            [calendar.month_abbr[m] for m in diff.index],
            rotation=0,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_diff)
# Reset compare_mode if locations are not exactly 2
elif st.session_state.compare_mode and len(st.session_state.locations) != 2:
    st.session_state.compare_mode = False
