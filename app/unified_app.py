import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# For interactive map
from streamlit_folium import st_folium
import folium

# Configure page before anything else
st.set_page_config(page_title="Historical Climate Data Explorer", layout="wide")

# SIMPLIFIED CSS APPROACH - focus on specific elements only
st.markdown(
    """
    <style>
    /* Set default padding for all elements */
    .stApp {
        --default-gap: 0.5rem;
    }
    
    /* Target the specific elements that cause the gap */
    .stApp > section > div > div:nth-of-type(1) > div:nth-of-type(1) > div,
    .element-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    /* Fix map container padding */
    .element-container:has(iframe) {
        padding-bottom: 0 !important;
        margin-bottom: -1.5rem !important;
    }
    
    /* Make buttons more compact */
    .stButton > button {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
    }
    
    /* Add styling for the tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #e6f0fd !important;
        border-bottom: 2px solid #4285F4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title
st.title("Historical Climate Data Explorer")

# Main app tabs
tab1, tab2 = st.tabs(["Wind Direction Analysis", "Temperature Comparison"])

# --- State Initialization ---
# Wind Direction App
if "location" not in st.session_state:
    # Set Heathrow as default pre-selected location
    st.session_state.location = (51.4700, -0.4543)  # Heathrow Airport
if "analyze_mode" not in st.session_state:
    # Auto-analyze by default for better UX
    st.session_state.analyze_mode = True
if "last_clicked_coords" not in st.session_state:
    st.session_state.last_clicked_coords = None
if "wind_data_cache" not in st.session_state:
    st.session_state.wind_data_cache = {}
    
# Temperature Comparison App
if "locations" not in st.session_state:
    st.session_state.locations = []
if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = False
if "temp_data_cache" not in st.session_state:
    st.session_state.temp_data_cache = {}

# Define common helper functions
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


def fetch_and_process_temp(lat, lon):
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
        return None, "Failed to fetch temperature data for this location."


### WIND DIRECTION APP ###
with tab1:
    # Default location: Heathrow Airport, UK
    heathrow_lat = 51.4700
    heathrow_lon = -0.4543

    st.markdown("### Select a location to analyze wind direction patterns")
    
    # --- Map Display and Interaction ---
    map_center = [heathrow_lat, heathrow_lon] if not st.session_state.location else list(st.session_state.location)
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

    # Render map with reduced height to minimize gap
    map_output = st_folium(m, width="100%", height=460, key="wind_map")

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
                
                # Enable analysis for the new location
                st.session_state.analyze_mode = True
                st.rerun()

    # Simple control section with reduced spacing
    # Add a small negative margin to pull controls closer to map
    st.markdown('<div style="margin-top:-2rem;"></div>', unsafe_allow_html=True)

    # Simple columns layout using native Streamlit
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Show selected location
        if st.session_state.location:
            lat, lon = st.session_state.location
            st.markdown(f"**Selected Location:** lat={lat:.4f}, lon={lon:.4f}")
        else:
            st.markdown("**No location selected**")
            
    with col2:
        # Use a small button with native Streamlit
        if st.button("Reset to Heathrow", key="reset_btn", use_container_width=True):
            st.session_state.location = (heathrow_lat, heathrow_lon)
            st.session_state.analyze_mode = True
            st.rerun()
            
    with col3:
        # Use a small button with native Streamlit
        if st.button(
            "Analyze Wind Direction", 
            key="analyze_btn",
            disabled=st.session_state.location is None,
            use_container_width=True
        ):
            st.session_state.analyze_mode = True
            st.rerun()

    # Wind analysis section
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
            wind_tab1, wind_tab2, wind_tab3 = st.tabs(["Westerly Wind Percentage", "E/W Monthly Stats", "Longest Wind Streaks"])
            
            with wind_tab1:
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
            
            with wind_tab2:
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
                
                # Create yearly heatmap for easterly wind preference
                st.markdown("### Easterly Wind Percentage by Day and Month")
                st.markdown("This heatmap shows the average percentage of easterly winds (vs westerly) for each day of each month across all years in the dataset.")
                
                # We need to make a new call to fetch data for this analysis
                end_date = datetime.today()
                start_year = end_date.year - 10  # Get 10 years of data
                start_date = datetime(start_year, 1, 1)
                
                # Fetch the data directly
                with st.spinner("Generating day-month heatmaps..."):
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
                        
                        # Process the data
                        df_winds = pd.DataFrame(data)
                        df_winds["time"] = pd.to_datetime(df_winds["time"])
                        df_winds["wind_direction_compass"] = df_winds["wind_direction_10m"].apply(deg_to_compass)
                        df_winds["year"] = df_winds["time"].dt.year
                        df_winds["month_num"] = df_winds["time"].dt.month
                        df_winds["day"] = df_winds["time"].dt.day
                        
                        # Filter for E/W winds only
                        df_winds = df_winds[df_winds["wind_direction_compass"].isin(["E", "W"])]
                        
                        # Calculate daily percentages for all years combined
                        daily_ew = df_winds.groupby(['month_num', 'day', 'wind_direction_compass']).size().unstack(fill_value=0)
                        daily_ew = daily_ew.reindex(columns=['E', 'W'], fill_value=0)
                        daily_total = daily_ew.sum(axis=1)
                        daily_percent = daily_ew.div(daily_total, axis=0) * 100
                        daily_percent = daily_percent.reset_index()
                        
                        # Create the pivot table for the heatmap (E percentage)
                        yearly_heatmap = daily_percent.pivot(index='month_num', columns='day', values='E')
                        
                        # Create the heatmap
                        plt.figure(figsize=(14, 8))
                        ax = sns.heatmap(
                            yearly_heatmap,
                            annot=True,
                            fmt=".1f",
                            cmap="RdYlGn_r",  # Reversed to show red for higher easterly %
                            cbar_kws={"label": "% Easterly Winds"},
                            linewidths=0.5,
                            linecolor="gray",
                            vmin=0,
                            vmax=100,
                            annot_kws={"size": 8},
                            mask=yearly_heatmap.isna()  # Explicitly mask NA values
                        )
                        plt.title("Easterly Wind Percentage by Day and Month (All Years)")
                        plt.xlabel("Day")
                        plt.ylabel("Month")
                        
                        # Set y-tick labels as month names
                        month_names = [calendar.month_abbr[m] for m in yearly_heatmap.index]
                        ax.set_yticklabels(month_names, rotation=0)
                        
                        plt.tight_layout()
                        st.pyplot(plt)
                        
                        # Current year heatmap
                        st.markdown("### Easterly Wind Percentage by Day and Month (Current Year Only)")
                        st.markdown("This heatmap shows the percentage of easterly winds for each day of each month for the current year only.")
                        
                        # Filter for current year data
                        current_year = datetime.now().year
                        df_current_year = df_winds[df_winds['year'] == current_year].copy()
                        
                        if not df_current_year.empty:
                            # Calculate daily percentages for current year
                            current_daily_ew = df_current_year.groupby(['month_num', 'day', 'wind_direction_compass']).size().unstack(fill_value=0)
                            current_daily_ew = current_daily_ew.reindex(columns=['E', 'W'], fill_value=0)
                            current_daily_total = current_daily_ew.sum(axis=1)
                            
                            # Add a column to track if there were any wind observations at all
                            current_daily_has_winds = current_daily_total > 0
                            
                            current_daily_percent = current_daily_ew.div(current_daily_total, axis=0) * 100
                            current_daily_percent = current_daily_percent.reset_index()
                            
                            # Create the pivot table for the current year heatmap (E percentage)
                            current_yearly_heatmap = current_daily_percent.pivot(index='month_num', columns='day', values='E')
                            
                            # Create a summary of data availability
                            st.markdown("### Data Availability for Current Year")
                            # Count available vs missing days per month
                            total_days = df_current_year['day'].nunique()
                            missing_days = 365 - total_days  # Approximate for simplicity
                            st.write(f"Total days with wind direction data: **{total_days}** (Missing: ~{missing_days} days)")
                            
                            # Create a dataframe of month-by-month data coverage
                            days_by_month = df_current_year.groupby('month_num')['day'].nunique()
                            days_in_month = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}  # Simplified
                            coverage = pd.DataFrame({
                                'Month': [calendar.month_name[m] for m in days_by_month.index],
                                'Days with Data': days_by_month.values,
                                'Total Days': [days_in_month.get(m, 30) for m in days_by_month.index],
                            })
                            coverage['Coverage %'] = (coverage['Days with Data'] / coverage['Total Days'] * 100).round(1)
                            st.write(coverage)
                            
                            # Explanation for missing values
                            st.markdown("**Note:** White cells in the heatmap represent days with no E/W wind data available. This could be due to:")
                            st.markdown("- Missing data in the weather archive")
                            st.markdown("- Days with only N/S winds (no E/W component)")
                            st.markdown("- Days with incomplete observations")
                            
                            # Create the heatmap
                            plt.figure(figsize=(14, 8))
                            ax = sns.heatmap(
                                current_yearly_heatmap,
                                annot=True,
                                fmt=".1f",
                                cmap="RdYlGn_r",  # Reversed to show red for higher easterly %
                                cbar_kws={"label": "% Easterly Winds"},
                                linewidths=0.5,
                                linecolor="gray",
                                vmin=0,
                                vmax=100,
                                annot_kws={"size": 8},
                                mask=current_yearly_heatmap.isna()  # Explicitly mask NA values
                            )
                            plt.title(f"Easterly Wind Percentage by Day and Month ({current_year})")
                            plt.xlabel("Day")
                            plt.ylabel("Month")
                            
                            # Set y-tick labels as month names
                            month_names = [calendar.month_abbr[m] for m in current_yearly_heatmap.index]
                            ax.set_yticklabels(month_names, rotation=0)
                            
                            plt.tight_layout()
                            st.pyplot(plt)
                        else:
                            st.info(f"No data available for the current year ({current_year}).")
                    else:
                        st.error("Failed to fetch data for heatmap visualization.")
                
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
            
            with wind_tab3:
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

### TEMPERATURE COMPARISON APP ###
with tab2:
    # Default location: Milan, Italy
    milan_lat = 45.4642
    milan_lon = 9.19

    st.markdown("### Select two locations to compare apparent temperature")
    
    # --- Map Display and Interaction ---
    temp_map_center = [milan_lat, milan_lon]
    temp_m = folium.Map(location=temp_map_center, zoom_start=4, tiles="CartoDB Positron")

    # Add markers for existing locations
    for i, (lat, lon) in enumerate(st.session_state.locations):
        color = "blue" if i == 0 else "red"
        folium.Marker(
            location=[lat, lon],
            popup=f"Location {i+1}: {lat:.4f}, {lon:.4f}",
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(temp_m)

    temp_m.add_child(folium.LatLngPopup())

    # Render map and get output
    temp_map_output = st_folium(temp_m, width="100%", height=500, key="temp_map")

    # Check if map has TRULY changed
    if temp_map_output and temp_map_output.get("last_clicked"):
        current_click = temp_map_output.get("last_clicked")

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

    # Coordinates and buttons in a single row below the map
    temp_coord_col, temp_btn_col1, temp_btn_col2 = st.columns([2, 1, 2])
    with temp_coord_col:
        # Show selected locations
        for i, (lat, lon) in enumerate(st.session_state.locations):
            st.markdown(f"**Location {i+1}:** lat={lat:.4f}, lon={lon:.4f}")
    with temp_btn_col1:
        if st.button("Clear Locations", key="clear_locations_btn"):
            st.session_state.locations = []
            st.session_state.compare_mode = False
            # Clear data cache when locations are cleared
            st.session_state.temp_data_cache = {}
            st.rerun()
    with temp_btn_col2:
        if st.button(
            "Compare Apparent Temperature",
            disabled=len(st.session_state.locations) != 2,
            key="compare_temp_btn"
        ):
            st.session_state.compare_mode = True
            st.rerun()

    # Only show temperature comparison if compare_mode is True and there are exactly 2 locations
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
                    temp1, err1 = fetch_and_process_temp(lat1, lon1)
                    st.session_state.temp_data_cache[loc1_key] = (temp1, err1)
                else:
                    temp1, err1 = st.session_state.temp_data_cache[loc1_key]

                if loc2_key not in st.session_state.temp_data_cache:
                    temp2, err2 = fetch_and_process_temp(lat2, lon2)
                    st.session_state.temp_data_cache[loc2_key] = (temp2, err2)
                else:
                    temp2, err2 = st.session_state.temp_data_cache[loc2_key]
        else:
            # Use cached data
            temp1, err1 = st.session_state.temp_data_cache[loc1_key]
            temp2, err2 = st.session_state.temp_data_cache[loc2_key]
            
        st.markdown("## Temperature Comparison Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### Location 1: lat={lat1:.4f}, lon={lon1:.4f}")
            if err1:
                st.info(err1)
            else:
                # Define custom colormap for temperature
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
