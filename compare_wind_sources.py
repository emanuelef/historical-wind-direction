#!/usr/bin/env python3
"""
Compare the Heathrow official wind direction data with our Open-Meteo API results.

This script loads the wind direction data from both sources and creates comparative
visualizations to validate our analysis against the official data.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Constants
HEATHROW_DATA_DIR = "heathrow_data"
OPENMETEO_DATA_DIR = "openmeteo_data"
OUTPUT_DIR = "comparison_results"

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_heathrow_data():
    """
    Load the latest Heathrow wind direction data.
    
    Returns:
        tuple: (monthly_df, annual_df)
    """
    if not os.path.exists(HEATHROW_DATA_DIR):
        print(f"Error: Heathrow data directory ({HEATHROW_DATA_DIR}) not found.")
        return None, None
    
    # Find the latest files
    monthly_files = [f for f in os.listdir(HEATHROW_DATA_DIR) if f.startswith('heathrow_monthly_wind_') and f.endswith('.csv')]
    annual_files = [f for f in os.listdir(HEATHROW_DATA_DIR) if f.startswith('heathrow_annual_wind_') and f.endswith('.csv')]
    
    monthly_df = None
    annual_df = None
    
    if monthly_files:
        latest_monthly = sorted(monthly_files)[-1]
        try:
            monthly_df = pd.read_csv(os.path.join(HEATHROW_DATA_DIR, latest_monthly))
            print(f"Loaded monthly data: {latest_monthly}")
        except Exception as e:
            print(f"Error loading monthly data: {e}")
    
    if annual_files:
        latest_annual = sorted(annual_files)[-1]
        try:
            annual_df = pd.read_csv(os.path.join(HEATHROW_DATA_DIR, latest_annual))
            print(f"Loaded annual data: {latest_annual}")
        except Exception as e:
            print(f"Error loading annual data: {e}")
    
    return monthly_df, annual_df

def load_openmeteo_data():
    """
    Load or generate Open-Meteo API data for Heathrow location.
    
    Returns:
        tuple: (monthly_df, annual_df) Wind direction data from Open-Meteo
    """
    # Heathrow coordinates
    lat = 51.4700
    lon = -0.4543
    
    # Check if we already have cached data
    os.makedirs(OPENMETEO_DATA_DIR, exist_ok=True)
    
    # Look for any CSV files in the directory
    monthly_files = [f for f in os.listdir(OPENMETEO_DATA_DIR) if f.startswith('heathrow_openmeteo_monthly_') and f.endswith('.csv')]
    annual_files = [f for f in os.listdir(OPENMETEO_DATA_DIR) if f.startswith('heathrow_openmeteo_annual_') and f.endswith('.csv')]
    
    monthly_df = None
    annual_df = None
    
    if monthly_files:
        latest_monthly = sorted(monthly_files)[-1]
        try:
            monthly_path = os.path.join(OPENMETEO_DATA_DIR, latest_monthly)
            print(f"Loading Open-Meteo monthly data from {monthly_path}")
            monthly_df = pd.read_csv(monthly_path)
            
            # Add month name to improve data alignment
            month_names = {
                1: "January", 2: "February", 3: "March", 4: "April", 
                5: "May", 6: "June", 7: "July", 8: "August",
                9: "September", 10: "October", 11: "November", 12: "December"
            }
            monthly_df['month'] = monthly_df['month_num'].map(month_names)
        except Exception as e:
            print(f"Error loading Open-Meteo monthly data: {e}")
    
    if annual_files:
        latest_annual = sorted(annual_files)[-1]
        try:
            annual_path = os.path.join(OPENMETEO_DATA_DIR, latest_annual)
            print(f"Loading Open-Meteo annual data from {annual_path}")
            annual_df = pd.read_csv(annual_path)
        except Exception as e:
            print(f"Error loading Open-Meteo annual data: {e}")
    
    if monthly_df is None and annual_df is None:
        # If we don't have cached data, we would need to fetch it using our wind_direction_app
        print("No cached Open-Meteo data found. Please run the wind_direction_app.py first")
        print(f"with Heathrow coordinates (lat={lat}, lon={lon}) and save the results.")
        
    return monthly_df, annual_df

def align_data_for_comparison(heathrow_data, openmeteo_data):
    """
    Align the data from both sources for proper comparison.
    
    Args:
        heathrow_data (DataFrame): Heathrow official data
        openmeteo_data (DataFrame): Data from Open-Meteo API
        
    Returns:
        DataFrame: Aligned data for comparison
    """
    if heathrow_data is None or openmeteo_data is None:
        print("Cannot align data: missing input dataframes")
        return None
    
    print("Aligning data from both sources...")
    
    # Inspect the data structures
    print("\nHeathrow Data Structure:")
    print(heathrow_data.head())
    
    print("\nOpen-Meteo Data Structure:")
    print(openmeteo_data.head())
    
    # Reset index if it's a MultiIndex
    if isinstance(heathrow_data.index, pd.MultiIndex):
        heathrow_data = heathrow_data.reset_index()
    
    # Create a clean working copy
    heathrow = heathrow_data.copy()
    openmeteo = openmeteo_data.copy()
    
    # Create a synthetic comparison dataframe
    comparison_df = pd.DataFrame()
    
    try:
        # Standardize column names
        if 'year' not in heathrow.columns and 'Year' in heathrow.columns:
            heathrow = heathrow.rename(columns={'Year': 'year'})
            
        if 'westerly' not in heathrow.columns and 'W' in heathrow.columns:
            heathrow = heathrow.rename(columns={'W': 'westerly'})
            
        if 'easterly' not in heathrow.columns and 'E' in heathrow.columns:
            heathrow = heathrow.rename(columns={'E': 'easterly'})
            
        # Add the Heathrow data to the comparison
        comparison_df['year'] = heathrow['year']
        
        # If we have month data, add it
        if 'month' in heathrow.columns:
            comparison_df['month'] = heathrow['month']
            comparison_df['month_num'] = heathrow['month_num'] if 'month_num' in heathrow.columns else pd.NA
            
        # Add the westerly/easterly data
        if 'westerly' in heathrow.columns:
            comparison_df['heathrow_westerly'] = heathrow['westerly']
        elif 'W' in heathrow.columns:
            comparison_df['heathrow_westerly'] = heathrow['W']
        else:
            print("Warning: No westerly wind percentage found in Heathrow data")
            
        if 'easterly' in heathrow.columns:
            comparison_df['heathrow_easterly'] = heathrow['easterly']
        elif 'E' in heathrow.columns:
            comparison_df['heathrow_easterly'] = heathrow['E']
            
        # Now try to match up the Open-Meteo data
        # For monthly data
        if 'month' in comparison_df.columns:
            # Create a new key based on year and month
            if 'month_num' in openmeteo.columns:
                # Find matching records in openmeteo dataframe
                openmeteo_values = []
                
                for _, row in comparison_df.iterrows():
                    year = row['year']
                    if 'month_num' in row and not pd.isna(row['month_num']):
                        month_num = row['month_num']
                    else:
                        # Convert month name to number if needed
                        month_map = {
                            'January': 1, 'February': 2, 'March': 3, 'April': 4,
                            'May': 5, 'June': 6, 'July': 7, 'August': 8,
                            'September': 9, 'October': 10, 'November': 11, 'December': 12
                        }
                        month_num = month_map.get(row['month'], None)
                        
                    # Find matching record
                    match = openmeteo[(openmeteo['year'] == year) & (openmeteo['month_num'] == month_num)]
                    
                    if len(match) > 0:
                        openmeteo_values.append(match['W'].values[0])
                    else:
                        openmeteo_values.append(np.nan)
                
                comparison_df['openmeteo_westerly'] = openmeteo_values
        
        # For annual data (simpler)
        elif 'year' in comparison_df.columns and 'year' in openmeteo.columns:
            # Create a lookup dictionary
            openmeteo_lookup = dict(zip(openmeteo['year'], openmeteo['W']))
            
            # Map the values
            comparison_df['openmeteo_westerly'] = comparison_df['year'].map(openmeteo_lookup)
        
        # Calculate difference if we have both sources
        if 'heathrow_westerly' in comparison_df.columns and 'openmeteo_westerly' in comparison_df.columns:
            comparison_df['difference'] = comparison_df['heathrow_westerly'] - comparison_df['openmeteo_westerly']
        
        return comparison_df
            
    except Exception as e:
        print(f"Error during data alignment: {e}")
        
        # Fallback to simple placeholder
        comparison_df = pd.DataFrame({
            'year': heathrow_data['year'] if 'year' in heathrow_data.columns else heathrow_data.index,
            'heathrow_westerly': np.nan,
            'openmeteo_westerly': np.nan,
            'difference': np.nan
        })
        
        return comparison_df

def create_comparison_visualizations(comparison_df, heathrow_annual):
    """
    Create visualizations comparing the data sources.
    
    Args:
        comparison_df (DataFrame): Aligned comparison data
        heathrow_annual (DataFrame): Heathrow annual data
    """
    ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d")
    
    if comparison_df is not None:
        print("Creating comparison visualizations...")
        
        # Create a comparison visualization if we have valid data
        if 'heathrow_westerly' in comparison_df.columns and 'openmeteo_westerly' in comparison_df.columns:
            # Filter to only include rows with both data sources
            valid_comparison = comparison_df.dropna(subset=['heathrow_westerly', 'openmeteo_westerly'])
            
            if len(valid_comparison) > 0:
                print(f"Found {len(valid_comparison)} valid data points for visualization")
                
                # Convert to numeric if needed
                valid_comparison['heathrow_westerly'] = pd.to_numeric(valid_comparison['heathrow_westerly'], errors='coerce')
                valid_comparison['openmeteo_westerly'] = pd.to_numeric(valid_comparison['openmeteo_westerly'], errors='coerce')
                
                # Bar chart comparison by year
                plt.figure(figsize=(14, 7))
                plt.title("Comparison of Westerly Wind Percentage by Year")
                
                # Sort by year for better visualization
                valid_comparison = valid_comparison.sort_values('year')
                
                # Create bar chart - grouped by year
                years = valid_comparison['year'].unique()
                x = np.arange(len(years))
                width = 0.35
                
                # Aggregate data by year if there are multiple entries per year
                yearly_data = valid_comparison.groupby('year').agg({
                    'heathrow_westerly': 'mean',
                    'openmeteo_westerly': 'mean'
                }).reset_index()
                
                # Plot bars
                plt.bar(x - width/2, yearly_data['heathrow_westerly'], width, label='Heathrow Official', color='blue', alpha=0.7)
                plt.bar(x + width/2, yearly_data['openmeteo_westerly'], width, label='Open-Meteo API', color='red', alpha=0.7)
                
                plt.xlabel("Year")
                plt.ylabel("Westerly Wind %")
                plt.xticks(x, yearly_data['year'].astype(str))
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(OUTPUT_DIR, f"wind_comparison_bar_{timestamp}.png"))
                plt.close()
                
                # Scatter plot
                plt.figure(figsize=(10, 10))
                plt.title("Correlation between Data Sources")
                plt.scatter(valid_comparison['heathrow_westerly'], valid_comparison['openmeteo_westerly'], 
                            alpha=0.7)
                plt.xlabel("Heathrow Official Westerly %")
                plt.ylabel("Open-Meteo Westerly %")
                
                # Add diagonal line (perfect correlation)
                max_val = max(valid_comparison['heathrow_westerly'].max(), valid_comparison['openmeteo_westerly'].max())
                min_val = min(valid_comparison['heathrow_westerly'].min(), valid_comparison['openmeteo_westerly'].min())
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                
                # Add correlation value to plot
                corr = valid_comparison['heathrow_westerly'].corr(valid_comparison['openmeteo_westerly'])
                plt.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=plt.gca().transAxes,
                        bbox=dict(facecolor='white', alpha=0.8))
                
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(OUTPUT_DIR, f"wind_comparison_scatter_{timestamp}.png"))
                plt.close()
                
                # Difference histogram
                if 'difference' in valid_comparison.columns:
                    plt.figure(figsize=(12, 6))
                    plt.title("Histogram of Differences (Heathrow - Open-Meteo)")
                    plt.hist(valid_comparison['difference'], bins=15, alpha=0.7, color='darkblue')
                    plt.xlabel("Difference in Westerly Wind % (Heathrow - Open-Meteo)")
                    plt.ylabel("Frequency")
                    plt.grid(True, alpha=0.3)
                    plt.axvline(x=0, color='r', linestyle='--')
                    plt.savefig(os.path.join(OUTPUT_DIR, f"wind_comparison_diff_{timestamp}.png"))
                    plt.close()
                
                # Time series if we have date data
                if 'month' in valid_comparison.columns and 'year' in valid_comparison.columns:
                    try:
                        # Try to create a date column
                        if 'month_num' in valid_comparison.columns:
                            valid_comparison['date'] = pd.to_datetime(
                                valid_comparison['year'].astype(str) + '-' + 
                                valid_comparison['month_num'].astype(str) + '-01'
                            )
                        else:
                            # Convert month names to numbers
                            month_map = {
                                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                                'September': 9, 'October': 10, 'November': 11, 'December': 12
                            }
                            
                            # Create month number column
                            valid_comparison['month_num'] = valid_comparison['month'].map(month_map)
                            valid_comparison['date'] = pd.to_datetime(
                                valid_comparison['year'].astype(str) + '-' + 
                                valid_comparison['month_num'].astype(str).fillna('1') + '-01'
                            )
                        
                        valid_comparison = valid_comparison.sort_values('date')
                        
                        # Line chart comparison
                        plt.figure(figsize=(14, 7))
                        plt.title("Time Series of Westerly Wind Percentage")
                        plt.plot(valid_comparison['date'], valid_comparison['heathrow_westerly'], 
                                'b-', label='Heathrow Official')
                        plt.plot(valid_comparison['date'], valid_comparison['openmeteo_westerly'], 
                                'r--', label='Open-Meteo API')
                        plt.xlabel("Date")
                        plt.ylabel("Westerly Wind %")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(OUTPUT_DIR, f"wind_comparison_time_{timestamp}.png"))
                        plt.close()
                    except Exception as e:
                        print(f"Error creating time series plot: {e}")
            else:
                print("No valid overlapping data points found for visualization")
                
                # Create a placeholder
                plt.figure(figsize=(10, 6))
                plt.title("Comparison of Wind Direction Data Sources")
                plt.text(0.5, 0.5, "Insufficient overlapping data\nfor meaningful comparison", 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"wind_comparison_{timestamp}.png"))
                plt.close()
        else:
            print("Required columns not found for comparison visualization")
            
            # Create a placeholder
            plt.figure(figsize=(10, 6))
            plt.title("Comparison of Wind Direction Data Sources")
            plt.text(0.5, 0.5, "Missing required data columns\nfor comparison visualization", 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"wind_comparison_{timestamp}.png"))
            plt.close()
    else:
        print("No comparison data available for visualization")
        
        # Create a placeholder
        plt.figure(figsize=(10, 6))
        plt.title("Comparison of Wind Direction Data Sources")
        plt.text(0.5, 0.5, "No comparison data available", 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"wind_comparison_{timestamp}.png"))
        plt.close()
        
    print(f"Comparison visualization saved to {OUTPUT_DIR}")

def calculate_statistics(comparison_df):
    """
    Calculate statistical measures of agreement between data sources.
    
    Args:
        comparison_df (DataFrame): Aligned comparison data
        
    Returns:
        dict: Statistics about the comparison
    """
    if comparison_df is None:
        return None
    
    print("Calculating comparison statistics...")
    
    # Check if we have the necessary columns
    if 'heathrow_westerly' in comparison_df.columns and 'openmeteo_westerly' in comparison_df.columns:
        # Remove rows with missing data
        valid_data = comparison_df.dropna(subset=['heathrow_westerly', 'openmeteo_westerly'])
        
        if len(valid_data) > 0:
            # Calculate correlation
            correlation = valid_data['heathrow_westerly'].corr(valid_data['openmeteo_westerly'])
            
            # Calculate RMSE
            rmse = np.sqrt(((valid_data['heathrow_westerly'] - valid_data['openmeteo_westerly']) ** 2).mean())
            
            # Calculate mean difference
            mean_diff = (valid_data['heathrow_westerly'] - valid_data['openmeteo_westerly']).mean()
            
            # Calculate mean absolute difference
            mean_abs_diff = np.abs(valid_data['heathrow_westerly'] - valid_data['openmeteo_westerly']).mean()
            
            # Additional statistics
            max_diff = np.abs(valid_data['heathrow_westerly'] - valid_data['openmeteo_westerly']).max()
            agreement_count = sum(np.abs(valid_data['heathrow_westerly'] - valid_data['openmeteo_westerly']) < 5)
            agreement_pct = agreement_count / len(valid_data) * 100
            
            stats = {
                'correlation': correlation,
                'rmse': rmse,
                'mean_difference': mean_diff,
                'mean_absolute_difference': mean_abs_diff,
                'max_absolute_difference': max_diff,
                'agreement_within_5pct': f"{agreement_pct:.1f}%",
                'sample_size': len(valid_data)
            }
            
            return stats
        else:
            print("No valid overlapping data points found for statistics calculation")
    else:
        print("Required columns not found in comparison data")
    
    # Return placeholder if calculations couldn't be performed
    return {
        'correlation': np.nan,
        'rmse': np.nan,
        'mean_difference': np.nan,
        'status': 'Could not calculate - missing data'
    }

def main():
    """Main function to run the comparison."""
    print("Heathrow Wind Data Comparison Tool")
    print("=================================")
    
    # Load the Heathrow official data
    heathrow_monthly, heathrow_annual = load_heathrow_data()
    
    if heathrow_monthly is None and heathrow_annual is None:
        print("Error: No Heathrow official data available.")
        print("Please run heathrow_wind_scraper.py first.")
        return
    
    # Load or generate Open-Meteo API data
    openmeteo_monthly, openmeteo_annual = load_openmeteo_data()
    
    if openmeteo_monthly is None and openmeteo_annual is None:
        print("Error: No Open-Meteo data available.")
        return
    
    # Align data for comparison
    # Use monthly data if available, otherwise try annual
    if heathrow_monthly is not None and openmeteo_monthly is not None:
        print("\nPerforming monthly data comparison...")
        comparison_df = align_data_for_comparison(heathrow_monthly, openmeteo_monthly)
    elif heathrow_annual is not None and openmeteo_annual is not None:
        print("\nPerforming annual data comparison...")
        comparison_df = align_data_for_comparison(heathrow_annual, openmeteo_annual)
    else:
        print("\nError: Incompatible data formats between sources.")
        return
    
    # Calculate comparison statistics
    comparison_stats = calculate_statistics(comparison_df)
    if comparison_stats:
        print("\nComparison Statistics:")
        for stat_name, stat_value in comparison_stats.items():
            print(f"- {stat_name}: {stat_value}")
    
    # Create comparison visualizations
    create_comparison_visualizations(comparison_df, heathrow_annual)
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main()
