#!/usr/bin/env python3
"""
Heathrow Wind Direction Data Scraper

This script extracts wind direction data from the Heathrow Airport operational data page.
It can either scrape the data directly from the website or parse it from a local copy.
The script outputs CSV files for monthly and annual wind direction data.

Usage:
  python heathrow_wind_scraper.py
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from datetime import datetime

# Constants
URL = "https://www.heathrow.com/company/local-community/noise/data/operational-data"
OUTPUT_DIR = "heathrow_data"

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def scrape_website():
    """
    Attempt to scrape the data directly from the Heathrow website.
    
    Returns:
        tuple: (monthly_df, annual_df) or (None, None) if scraping fails
    """
    try:
        print("Attempting to scrape data from Heathrow website...")
        response = requests.get(URL, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This is a placeholder - actual implementation would depend on website structure
            # which may require inspection of the specific elements containing the tables
            
            print("Web scraping implementation is placeholder only.")
            print("Using fallback method with local data extraction.")
            return None, None
        else:
            print(f"Failed to access website: HTTP {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Error during web scraping: {str(e)}")
        return None, None

def extract_from_table_text(table_text):
    """
    Extract data from the provided table text.
    
    Args:
        table_text (str): Text containing the wind direction data tables
        
    Returns:
        tuple: (monthly_df, annual_df)
    """
    print("Extracting data from provided table text...")
    
    # Split into monthly and annual sections
    monthly_pattern = r"Monthly wind direction(.*?)Annual wind direction"
    monthly_match = re.search(monthly_pattern, table_text, re.DOTALL)
    
    annual_pattern = r"Annual wind direction(.*?)$"
    annual_match = re.search(annual_pattern, table_text, re.DOTALL)
    
    monthly_data = []
    annual_data = []
    
    if monthly_match:
        monthly_text = monthly_match.group(1)
        # Process monthly data - look for year and month patterns
        year_blocks = re.findall(r"(\d{4})\s+Westerly %\s+Easterly %\s+(.*?)(?=\d{4}|\Z)", monthly_text, re.DOTALL)
        
        for year, months_data in year_blocks:
            # Find all month entries
            month_entries = re.findall(r"(\w+)\s+(\d+)\s+(\d+)", months_data)
            for month_name, westerly, easterly in month_entries:
                monthly_data.append({
                    "year": year,
                    "month": month_name,
                    "westerly": int(westerly),
                    "easterly": int(easterly)
                })
    
    if annual_match:
        annual_text = annual_match.group(1)
        # Process annual data
        annual_entries = re.findall(r"(\d{4})\s+(\d+)\s+(\d+)", annual_text)
        
        for year, westerly, easterly in annual_entries:
            annual_data.append({
                "year": year,
                "westerly": int(westerly),
                "easterly": int(easterly)
            })
    
    # Convert to DataFrames
    monthly_df = pd.DataFrame(monthly_data)
    annual_df = pd.DataFrame(annual_data)
    
    # Add month number for sorting
    if not monthly_df.empty:
        month_to_num = {month: i for i, month in enumerate(calendar.month_name) if i > 0}
        monthly_df["month_num"] = monthly_df["month"].apply(lambda x: month_to_num.get(x.capitalize(), 0))
        monthly_df = monthly_df.sort_values(["year", "month_num"])
    
    return monthly_df, annual_df

def process_local_data():
    """
    Process the wind direction data from the text provided in this script.
    
    Returns:
        tuple: (monthly_df, annual_df)
    """
    # Wind direction data as provided in the script
    data_text = """
Wind direction
Below is a breakdown of the percentage of time we were on westerly and easterly operations.

Monthly wind direction
2025	Westerly %	Easterly %
January	80	20
February	47	53
March	44	56
April	29	71
May	38	62
June	85	15
 
2024	Westerly %	Easterly %
January	75	25
February	87	13
March	70	30
April	81	19
May	51	49
June	82	18
July	81	19
August	84	16
September	59	41
October	66	34
November	62	38
December	84	16
 
Annual wind direction
Year	Westerly %	Easterly %
2000	78	22
2001	70	30
2002	69	31
2003	63	37
2004	78	22
2005	72	28
2006	71	29
2007	73	27
2008	73	27
2009	75	25
2010	66	34
2011	71	29
2012	77	23
2013	67	33
2014	70	30
2015	73	27
2016	70	30
2017	81	19
2018	63	37
2019	75	25
2020	81	19
2021	71	29
2022	72	28
2023	71	29
2024	73	27
    """
    
    return extract_from_table_text(data_text)

def save_data(monthly_df, annual_df):
    """
    Save the extracted data to CSV files.
    
    Args:
        monthly_df (DataFrame): Monthly wind direction data
        annual_df (DataFrame): Annual wind direction data
    """
    ensure_output_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d")
    
    if not monthly_df.empty:
        monthly_file = os.path.join(OUTPUT_DIR, f"heathrow_monthly_wind_{timestamp}.csv")
        monthly_df.to_csv(monthly_file, index=False)
        print(f"Monthly data saved to {monthly_file}")
    
    if not annual_df.empty:
        annual_file = os.path.join(OUTPUT_DIR, f"heathrow_annual_wind_{timestamp}.csv")
        annual_df.to_csv(annual_file, index=False)
        print(f"Annual data saved to {annual_file}")

def create_visualizations(monthly_df, annual_df):
    """
    Create visualizations of the wind direction data.
    
    Args:
        monthly_df (DataFrame): Monthly wind direction data
        annual_df (DataFrame): Annual wind direction data
    """
    ensure_output_dir()
    
    # 1. Annual Wind Direction Trend
    if not annual_df.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(annual_df["year"], annual_df["westerly"], marker='o', linewidth=2, label="Westerly %")
        plt.plot(annual_df["year"], annual_df["easterly"], marker='s', linewidth=2, label="Easterly %")
        plt.title("Heathrow Annual Wind Direction (2000-2024)")
        plt.xlabel("Year")
        plt.ylabel("Percentage (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(annual_df["year"][::2], rotation=45)  # Show every other year
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "heathrow_annual_wind_trend.png"))
        plt.close()
        print("Annual trend visualization saved")
    
    # 2. Monthly heatmap for the most recent years
    if not monthly_df.empty:
        # Get the last 2 years of data
        years = sorted(monthly_df["year"].unique())[-2:]
        recent_data = monthly_df[monthly_df["year"].isin(years)].copy()
        
        # Create pivot table for heatmap
        heatmap_data = recent_data.pivot(index="month_num", columns="year", values="westerly")
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt="d",
            cmap="RdYlGn",
            cbar_kws={"label": "% Westerly Winds"},
            linewidths=0.5,
            vmin=0,
            vmax=100
        )
        plt.title(f"Heathrow Monthly Westerly Wind Percentage ({'-'.join(years)})")
        plt.ylabel("")
        plt.xlabel("")
        
        # Set month names as y-axis labels
        ax.set_yticklabels([calendar.month_abbr[m] for m in heatmap_data.index], rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "heathrow_monthly_heatmap.png"))
        plt.close()
        print("Monthly heatmap visualization saved")
        
    # 3. Compare with our model for the current year if available
    # This would be implemented if we have model data to compare against

def compare_with_model(heathrow_df):
    """
    Compare official Heathrow data with our model predictions.
    This is a placeholder function that would be implemented if we had model data.
    
    Args:
        heathrow_df (DataFrame): Official Heathrow wind direction data
    """
    # Placeholder for comparing official data with model predictions
    print("Model comparison not implemented yet.")

def main():
    """Main function to run the scraper."""
    print("Heathrow Wind Direction Data Scraper")
    print("====================================")
    
    # Try web scraping first (placeholder implementation)
    monthly_df, annual_df = scrape_website()
    
    # Fall back to local data extraction if scraping fails
    if monthly_df is None or annual_df is None:
        monthly_df, annual_df = process_local_data()
    
    if monthly_df.empty and annual_df.empty:
        print("Failed to extract wind direction data.")
        return
    
    # Print summary
    print("\nData Summary:")
    if not monthly_df.empty:
        print(f"- Monthly data: {len(monthly_df)} entries")
        print(f"- Years covered (monthly): {monthly_df['year'].min()}-{monthly_df['year'].max()}")
    
    if not annual_df.empty:
        print(f"- Annual data: {len(annual_df)} entries")
        print(f"- Years covered (annual): {annual_df['year'].min()}-{annual_df['year'].max()}")
    
    # Save to CSV files
    save_data(monthly_df, annual_df)
    
    # Create visualizations
    create_visualizations(monthly_df, annual_df)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
