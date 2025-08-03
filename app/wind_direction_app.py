import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import numpy as np

# For interactive map
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="Historical Wind Direction Explorer", layout="wide")

# Default location: Heathrow Airport, UK
latitude = 51.4700
longitude = -0.4543

st.markdown("### Select a location to analyze wind direction patterns")

# --- State Initialization ---
if "location" not in st.session_state:
    # Set Heathrow as default pre-selected location
    st.session_state.location = (latitude, longitude)
if "analyze_mode" not in st.session_state:
    # Auto-analyze by default for better UX
    st.session_state.analyze_mode = True
if "last_clicked_coords" not in st.session_state:
    st.session_state.last_clicked_coords = None
# Cache for wind data - dictionary structure: {loc_key: (fetch_date, data)}
if "wind_data_cache" not in st.session_state:
    st.session_state.wind_data_cache = {}
# Note: We're using per-location cache dates stored with each entry, not a global cache date

# --- Map Display and Interaction ---
map_center = [latitude, longitude]
m = folium.Map(location=map_center, zoom_start=4, tiles="CartoDB Positron")

# Add marker for the selected location
if st.session_state.location:
    lat, lon = st.session_state.location
    folium.Marker(
        location=[lat, lon],
        popup=f"Selected Location: {lat:.4f}, {lon:.4f}",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

m.add_child(folium.LatLngPopup())

# Render map and get output
map_output = st_folium(m, width="100%", height=500, key="map")

# Check if map has TRULY changed in a meaningful way (new clicks, not just zoom/pan)
if map_output and map_output.get("last_clicked"):
    current_click = map_output.get("last_clicked")

    # Only process click if it's new AND different from previous click
    if current_click != st.session_state.last_clicked_coords:
        lat, lon = current_click["lat"], current_click["lng"]

        # Save current clicked coordinates to compare next time
        st.session_state.last_clicked_coords = current_click

        # Check if the location is different from the current one
        is_new = True
        if st.session_state.location:
            curr_lat, curr_lon = st.session_state.location
            is_new = not (abs(lat - curr_lat) < 1e-4 and abs(lon - curr_lon) < 1e-4)

        if is_new:
            # This is a true new location selection
            st.session_state.location = (lat, lon)
            
            # No need to clear cache - we'll use date-based cache expiration instead
            
            # Enable analysis for the new location
            st.session_state.analyze_mode = True
            st.rerun()

# Remove vertical gap between map and controls
st.markdown(
    "<style>div[data-testid='stVerticalBlock']:has(div.folium-map) + div[data-testid='stVerticalBlock'] { margin-top: -32px !important; } </style>",
    unsafe_allow_html=True,
)

# Coordinates and buttons in a single row below the map
coord_col, btn_col1, btn_col2 = st.columns([2, 1, 2])
with coord_col:
    # Show selected location
    if st.session_state.location:
        lat, lon = st.session_state.location
        st.markdown(f"**Selected Location:** lat={lat:.4f}, lon={lon:.4f}")
    else:
        st.markdown("**No location selected**")
        
with btn_col1:
    if st.button("Reset to Heathrow"):
        # Reset to Heathrow default
        st.session_state.location = (latitude, longitude)
        st.session_state.analyze_mode = True
        st.rerun()
        
with btn_col2:
    if st.button(
        "Analyze Wind Direction",
        disabled=st.session_state.location is None,
    ):
        st.session_state.analyze_mode = True
        st.rerun()


def deg_to_compass(deg):
    """Convert degrees to compass direction."""
    if pd.isna(deg):
        return None
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = int((deg + 22.5) // 45) % 8
    return directions[ix]


def fetch_and_process_wind(lat, lon):
    end_date = datetime.today()
    start_year = end_date.year - 10  # Get 10 years of data
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
    response = requests.get(url, params=params)
    if response.status_code == 200 and "hourly" in response.json():
        data = response.json()["hourly"]
        if not data or not data.get("time") or len(data["time"]) == 0:
            return None, None, None, "No data available for this location and time range."
        
        df = pd.DataFrame(data)
        if df.empty:
            return None, None, None, "No data returned for this location."
        
        # Process the data
        df["time"] = pd.to_datetime(df["time"])
        df["wind_direction_compass"] = df["wind_direction_10m"].apply(deg_to_compass)
        df["year"] = df["time"].dt.year.astype(str)
        df["month_num"] = df["time"].dt.month
        df["date"] = df["time"].dt.date
        
        # Monthly analysis - E/W wind percentages
        ew_counts = (
            df[df["wind_direction_compass"].isin(["E", "W"])]
            .groupby(["month_num", "year", "wind_direction_compass"])
            .size()
            .unstack(fill_value=0)
        )
        ew_counts = ew_counts.reindex(columns=["E", "W"], fill_value=0)
        ew_total = ew_counts.sum(axis=1)
        ew_percent = ew_counts.div(ew_total, axis=0) * 100
        ew_percent = ew_percent.fillna(0)
        ew_percent = ew_percent.reset_index()
        
        # Create heatmap data: percentage of westerly wind by month and year
        heatmap_data = ew_percent.pivot(index="month_num", columns="year", values="W")
        heatmap_data = heatmap_data.sort_index()
        
        # Daily analysis - find longest consecutive days with E or W predominance
        daily_counts = (
            df[df["wind_direction_compass"].isin(["E", "W"])]
            .groupby(["date", "wind_direction_compass"])
            .size()
            .unstack(fill_value=0)
        )
        daily_counts = daily_counts.reindex(columns=["E", "W"], fill_value=0)
        daily_counts["predominant"] = daily_counts.apply(lambda row: "E" if row["E"] > row["W"] else "W" if row["W"] > row["E"] else "Equal", axis=1)
        
        # Find longest runs of E and W
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
            return runs[:5]  # Top 5 runs
        
        top_e_runs = find_longest_runs("E")
        top_w_runs = find_longest_runs("W")
        top_periods = pd.DataFrame(
            [("E", start, end, length) for start, end, length in top_e_runs]
            + [("W", start, end, length) for start, end, length in top_w_runs],
            columns=["direction", "start_date", "end_date", "days"],
        )
        top_periods = top_periods.sort_values(["direction", "days"], ascending=[True, False])
        
        return heatmap_data, ew_percent, top_periods, None
    else:
        return None, None, None, "Failed to fetch wind data for this location."


# Only show analysis if analyze_mode is True and there's a location
if st.session_state.analyze_mode and st.session_state.location is not None:
    # Get the selected location
    lat, lon = st.session_state.location
    loc_key = f"{lat:.4f}_{lon:.4f}"
    
    # Check if we need to fetch data based on cache status and date
    current_date = datetime.now().date().isoformat()
    
    need_fetch = False
    
    # Fetch if: 1) Location not in cache, or 2) Cache date has changed
    if loc_key not in st.session_state.wind_data_cache:
        need_fetch = True
    else:
        # Check if the cached data is from a different day
        cached_date = st.session_state.wind_data_cache[loc_key][0]
        if cached_date != current_date:
            need_fetch = True
        
    if need_fetch:
        with st.spinner("Fetching and processing wind data for the selected location..."):
            heatmap_data, ew_percent, top_periods, error = fetch_and_process_wind(lat, lon)
            st.session_state.wind_data_cache[loc_key] = (current_date, heatmap_data, ew_percent, top_periods, error)
    else:
        # Use cached data
        _, heatmap_data, ew_percent, top_periods, error = st.session_state.wind_data_cache[loc_key]
    
    st.markdown(f"## Wind Analysis for: lat={lat:.4f}, lon={lon:.4f}")
    
    if error:
        st.error(error)
    else:
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Westerly Wind Percentage", "E/W Monthly Stats", "Longest Wind Streaks"])
        
        with tab1:
            st.markdown("### Westerly Wind Percentage by Month and Year")
            st.markdown("""
            This heatmap shows the percentage of westerly winds out of all E/W winds for each month over the past years. 
            Higher values (green) indicate more westerly winds, while lower values (red) indicate more easterly winds.
            The rightmost column shows the average for each month, and the bottom row shows the average for each year.
            """)
            
            # Add monthly and yearly averages to heatmap_data
            # Make a copy to avoid modifying the cached data
            heatmap_with_avgs = heatmap_data.copy()
            
            # Calculate monthly averages (across years) - add as a new column
            monthly_avgs = heatmap_with_avgs.mean(axis=1, skipna=True).round(1)
            heatmap_with_avgs['Average'] = monthly_avgs
            
            # Calculate yearly averages (across months) - add as a new row
            yearly_avgs = heatmap_with_avgs.mean(axis=0, skipna=True).round(1)
            yearly_avgs_df = pd.DataFrame([yearly_avgs], index=[13])  # Use 13 to ensure it's after all months
            heatmap_with_avgs = pd.concat([heatmap_with_avgs, yearly_avgs_df])
            
            # Create the heatmap
            plt.figure(figsize=(14, 9))
            ax = sns.heatmap(
                heatmap_with_avgs,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn",
                cbar_kws={"label": "% Westerly Winds"},
                linewidths=0.5,
                linecolor="gray",
                vmin=0,
                vmax=100,
                annot_kws={"size": 9},
            )
            plt.title("Westerly Wind Percentage per Month per Year")
            plt.ylabel("")
            plt.xlabel("")
            
            # Adjust y-tick labels to include "Average" for the last row
            y_labels = [calendar.month_abbr[m] if m <= 12 else "Average" for m in heatmap_with_avgs.index]
            ax.set_yticklabels(y_labels, rotation=0)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
        
        with tab2:
            st.markdown("### Monthly E/W Wind Statistics")
            
            # Calculate averages by month across years
            monthly_avg = ew_percent.groupby('month_num').agg({'E': 'mean', 'W': 'mean'}).reset_index()
            monthly_avg['month'] = monthly_avg['month_num'].apply(lambda x: calendar.month_abbr[x])
            
            # Create bar chart of monthly averages
            plt.figure(figsize=(14, 6))
            width = 0.35
            x = np.arange(len(monthly_avg))
            plt.bar(x - width/2, monthly_avg['E'], width, label='Easterly', color='#d62728')
            plt.bar(x + width/2, monthly_avg['W'], width, label='Westerly', color='#2ca02c')
            plt.xlabel('Month')
            plt.ylabel('Average Percentage (%)')
            plt.title('Average E/W Wind Percentage by Month')
            plt.xticks(x, monthly_avg['month'])
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(plt)
            
            # Show raw data
            st.markdown("#### Raw Monthly Data")
            display_data = ew_percent.copy()
            display_data['month_name'] = display_data['month_num'].apply(lambda x: calendar.month_abbr[x])
            display_data = display_data[['year', 'month_name', 'E', 'W']]
            display_data = display_data.rename(columns={
                'year': 'Year',
                'month_name': 'Month', 
                'E': 'Easterly %', 
                'W': 'Westerly %'
            })
            st.dataframe(display_data.round(1), use_container_width=True)
        
        with tab3:
            st.markdown("### Longest Consecutive Days with E or W Wind Predominance")
            st.markdown("""
            This table shows periods when either easterly or westerly winds were consistently predominant.
            These streaks can indicate sustained weather patterns or seasonal effects.
            """)
            
            # Format the data for display
            display_periods = top_periods.copy()
            display_periods['start_date'] = pd.to_datetime(display_periods['start_date']).dt.strftime('%Y-%m-%d')
            display_periods['end_date'] = pd.to_datetime(display_periods['end_date']).dt.strftime('%Y-%m-%d')
            display_periods = display_periods.rename(columns={
                'direction': 'Wind Direction',
                'start_date': 'Start Date',
                'end_date': 'End Date',
                'days': 'Duration (days)'
            })
            st.dataframe(display_periods, use_container_width=True)
            
    # Additional wind analysis section
    st.markdown("## Yearly Wind Pattern Summary")
    
    # Calculate yearly predominant wind directions
    yearly_summary = ew_percent.groupby('year').agg({
        'E': 'mean', 
        'W': 'mean'
    }).reset_index()
    
    yearly_summary['Predominant'] = yearly_summary.apply(
        lambda row: 'Easterly' if row['E'] > row['W'] else 'Westerly' if row['W'] > row['E'] else 'Equal',
        axis=1
    )
    
    # Create bar chart of yearly wind patterns
    plt.figure(figsize=(14, 6))
    bars = plt.bar(
        yearly_summary['year'], 
        yearly_summary['W'],
        label='Westerly %'
    )
    
    # Color bars based on predominant direction
    for i, bar in enumerate(bars):
        if yearly_summary.iloc[i]['W'] > 50:
            bar.set_color('#2ca02c')  # Green for westerly
        else:
            bar.set_color('#d62728')  # Red for easterly
            
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% Threshold')
    plt.xlabel('Year')
    plt.ylabel('Westerly Wind (%)')
    plt.title('Yearly Westerly Wind Percentage')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    
    # Add a simple data table
    yearly_summary_display = yearly_summary.copy()
    yearly_summary_display.columns = ['Year', 'Easterly %', 'Westerly %', 'Predominant Direction']
    st.dataframe(yearly_summary_display.round(1), use_container_width=True)
# Reset analyze_mode if location is cleared
elif st.session_state.analyze_mode and st.session_state.location is None:
    st.session_state.analyze_mode = False
