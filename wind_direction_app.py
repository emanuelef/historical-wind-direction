#!/usr/bin/env python3
"""
Main script for the wind direction analysis project.

This script fetches historical wind direction data from Open-Meteo API for
the Heathrow Airport location, processes it, and analyzes the predominant
wind patterns.
"""

import requests
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define location (Heathrow Airport)
LATITUDE = 51.4700
LONGITUDE = -0.4543

# Define directory structure
OUTPUT_DIR = "openmeteo_data"


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def deg_to_compass(deg):
    """Convert wind direction in degrees to compass direction."""
    import math

    if deg is None or (isinstance(deg, float) and math.isnan(deg)):
        return None
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = int((deg + 22.5) // 45) % 8
    return directions[ix]


def fetch_openmeteo_data(years=10):
    """
    Fetch historical wind data from Open-Meteo API.
    
    Args:
        years (int): Number of years of historical data to fetch
        
    Returns:
        DataFrame: Processed wind direction data
    """
    # Define time range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    
    # Open-Meteo API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Parameters
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "wind_speed_10m,wind_direction_10m",
        "timezone": "auto",
    }
    
    print(f"Fetching {years} years of wind direction data from Open-Meteo API...")
    print(f"Location: Heathrow Airport (lat={LATITUDE}, lon={LONGITUDE})")
    
    # Fetch data
    response = requests.get(url, params=params)
    
    # Print the full URL
    print("Full request URL:", response.url)
    
    # Check if the response is successful
    if response.status_code != 200:
        print(f"Error: API returned status code {response.status_code}")
        return None
    
    data = response.json()
    
    if "hourly" not in data:
        print("Error: Expected 'hourly' data not found in API response")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data["hourly"])
    
    # Add compass direction column
    df["wind_direction_compass"] = df["wind_direction_10m"].apply(deg_to_compass)
    
    # Add date-related columns
    df["datetime"] = pd.to_datetime(df["time"])
    df["month"] = df["datetime"].dt.to_period("M")
    df["year"] = df["datetime"].dt.year
    df["month_num"] = df["datetime"].dt.month
    
    return df


def analyze_wind_directions(df):
    """
    Analyze wind directions from the dataframe.
    
    Args:
        df (DataFrame): Wind direction data
        
    Returns:
        tuple: (monthly_data, annual_data)
    """
    if df is None:
        return None, None
    
    # List of all compass directions
    all_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    
    # Group by month and wind direction, count occurrences
    monthly_grouped = (
        df.groupby(["year", "month_num", "wind_direction_compass"])
        .size()
        .unstack(fill_value=0)
    )
    
    # Ensure all directions are present as columns
    for d in all_directions:
        if d not in monthly_grouped.columns:
            monthly_grouped[d] = 0
    monthly_grouped = monthly_grouped[all_directions]  # order columns
    
    # Calculate percentages
    monthly_percent = monthly_grouped.div(monthly_grouped.sum(axis=1), axis=0) * 100
    
    # Reset index for easier handling
    monthly_percent = monthly_percent.reset_index()
    
    # Calculate E and W percentages directly from the original data
    # First, create a new column that simplifies the direction to E or W
    def simplify_direction(direction):
        if direction in ["NE", "E", "SE"]:
            return "E"
        elif direction in ["SW", "W", "NW"]:
            return "W"
        else:
            return None  # N and S are neither E nor W
    
    df["simplified_direction"] = df["wind_direction_compass"].apply(simplify_direction)
    df_ew = df[df["simplified_direction"].isin(["E", "W"])]
    
    # Monthly E/W analysis
    ew_monthly = (
        df_ew.groupby(["year", "month_num", "simplified_direction"])
        .size()
        .unstack(fill_value=0)
    )
    ew_monthly = ew_monthly.reindex(columns=["E", "W"], fill_value=0)
    ew_monthly_total = ew_monthly.sum(axis=1)
    ew_monthly_percent = ew_monthly.div(ew_monthly_total, axis=0) * 100
    ew_monthly_percent = ew_monthly_percent.fillna(0).reset_index()
    
    # Add a predominant column
    def predominant_ew(row):
        if row["E"] > row["W"]:
            return "E"
        elif row["W"] > row["E"]:
            return "W"
        else:
            return "Equal"
    
    ew_monthly_percent["predominant"] = ew_monthly_percent.apply(predominant_ew, axis=1)
    
    # Annual E/W analysis
    ew_annual = (
        df_ew.groupby(["year", "simplified_direction"])
        .size()
        .unstack(fill_value=0)
    )
    ew_annual = ew_annual.reindex(columns=["E", "W"], fill_value=0)
    ew_annual_total = ew_annual.sum(axis=1)
    ew_annual_percent = ew_annual.div(ew_annual_total, axis=0) * 100
    ew_annual_percent = ew_annual_percent.fillna(0).reset_index()
    ew_annual_percent["predominant"] = ew_annual_percent.apply(predominant_ew, axis=1)
    
    return ew_monthly_percent, ew_annual_percent


def save_results(monthly_data, annual_data):
    """
    Save the processed data to CSV files.
    
    Args:
        monthly_data (DataFrame): Monthly wind direction data
        annual_data (DataFrame): Annual wind direction data
    """
    ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d")
    
    if monthly_data is not None:
        monthly_file = os.path.join(OUTPUT_DIR, f"heathrow_openmeteo_monthly_{timestamp}.csv")
        monthly_data.to_csv(monthly_file, index=False)
        print(f"Monthly data saved to: {monthly_file}")
    
    if annual_data is not None:
        annual_file = os.path.join(OUTPUT_DIR, f"heathrow_openmeteo_annual_{timestamp}.csv")
        annual_data.to_csv(annual_file, index=False)
        print(f"Annual data saved to: {annual_file}")


def visualize_data(monthly_data, annual_data):
    """
    Create visualizations of the wind direction data.
    
    Args:
        monthly_data (DataFrame): Monthly wind direction data
        annual_data (DataFrame): Annual wind direction data
    """
    ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d")
    
    if monthly_data is not None:
        # Create pivot table for heatmap visualization
        monthly_pivot = monthly_data.pivot(index="year", columns="month_num", values="W")
        
        plt.figure(figsize=(14, 8))
        plt.title("Westerly Wind Percentage by Month and Year (Open-Meteo Data)")
        
        # Create heatmap
        im = plt.imshow(monthly_pivot, cmap='coolwarm', aspect='auto', vmin=0, vmax=100)
        plt.colorbar(im, label="Westerly Wind %")
        
        # Set axis labels
        plt.xlabel("Month")
        plt.ylabel("Year")
        
        # Set x-ticks to month names
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        plt.xticks(range(12), month_names)
        plt.yticks(range(len(monthly_pivot.index)), monthly_pivot.index)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"openmeteo_monthly_heatmap_{timestamp}.png"))
        plt.close()
    
    if annual_data is not None:
        plt.figure(figsize=(12, 6))
        plt.title("Annual Westerly Wind Percentage (Open-Meteo Data)")
        plt.bar(annual_data["year"].astype(str), annual_data["W"])
        plt.axhline(y=50, color='r', linestyle='--')
        plt.xlabel("Year")
        plt.ylabel("Westerly Wind %")
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"openmeteo_annual_bar_{timestamp}.png"))
        plt.close()


def main():
    """Main function to run the analysis."""
    print("Heathrow Wind Direction Analysis using Open-Meteo API")
    print("===================================================")
    
    # Fetch data
    wind_data = fetch_openmeteo_data(years=10)
    
    if wind_data is None:
        print("Error fetching data. Exiting.")
        return
    
    # Analyze data
    monthly_results, annual_results = analyze_wind_directions(wind_data)
    
    # Display results
    if monthly_results is not None:
        print("\nMonthly E/W Wind Direction Summary:")
        print(monthly_results.tail(12).to_string(index=False))
    
    if annual_results is not None:
        print("\nAnnual E/W Wind Direction Summary:")
        print(annual_results.to_string(index=False))
    
    # Save results
    save_results(monthly_results, annual_results)
    
    # Create visualizations
    visualize_data(monthly_results, annual_results)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
