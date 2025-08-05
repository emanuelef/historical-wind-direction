# Historical Climate Data Explorer

This project provides interactive tools to analyze historical climate data, including:

1. **Wind Direction Analysis**: Study the patterns of easterly vs westerly winds over time
2. **Temperature Comparison**: Compare apparent temperature between two locations

The application is built with Streamlit and uses Open-Meteo's historical weather data API.

## Features

### Wind Direction Analysis
- Analyze wind patterns for any global location
- View percentage of easterly vs westerly winds by month and year
- Identify the longest consecutive periods of consistent wind direction
- Visualize annual trends in wind direction

### Temperature Comparison
- Compare apparent temperatures between two locations
- View monthly maximum apparent temperatures across years
- See temperature differences between locations with heat maps

## Running the App

### Using Docker (Recommended)

# Pull the latest image
docker pull ghcr.io/emanuelef/historical-wind-direction/wind-direction-app:main

# Run the container
docker run -p 8501:8501 ghcr.io/emanuelef/historical-wind-direction/wind-direction-app:main

# The image supports both amd64 and arm64 architectures (including Apple Silicon M1/M2/M3)
```

Or build and run locally:

```bash
# Build the Docker image
docker build -t wind-direction-app .

# Run the container
docker run -p 8501:8501 wind-direction-app
```

Once running, access the application at http://localhost:8501:
1. Heathrow Airport's official operational dataset
2. Open-Meteo API historical weather data

## Project Overview

The goal of this analysis is to validate the reliability of using Open-Meteo API as an alternative data source for historical wind direction patterns at Heathrow Airport. This is useful for researchers, meteorologists, and analysts who need reliable wind data but may not have direct access to official airport measurements.

## Key Components

- **Data Extraction**: Scripts for retrieving data from both Heathrow's official sources and the Open-Meteo API
- **Data Processing**: Tools for standardizing and aligning time series data from different sources
- **Statistical Analysis**: Comprehensive statistical evaluation of the agreement between data sources
- **Visualization**: Various plots and charts to illustrate the relationships and differences
- **Interactive Application**: Streamlit app for exploring wind direction patterns at any location worldwide

## Key Findings

1. **Strong Correlation**: The analysis revealed a Pearson correlation coefficient of 0.969 between the two data sources, indicating an extremely strong linear relationship.

2. **Systematic Bias**: Heathrow data consistently reports approximately 4.1% higher westerly wind percentages compared to Open-Meteo data. This bias is statistically significant and should be accounted for when using Open-Meteo as an alternative.

3. **Seasonal Variation**: The agreement between sources shows some seasonal differences, with the strongest correlation in winter months.

4. **Statistical Significance**: Both parametric (paired t-test) and non-parametric (Wilcoxon) tests confirm that the differences between sources are statistically significant (p < 0.05).

5. **Agreement Metrics**: 
   - RMSE: 6.42%
   - 61.1% of measurements agree within 5% 
   - 83.3% of measurements agree within 10%

## Technical Details

The repository includes:

- **Python Scripts**:
  - `heathrow_wind_scraper.py`: Extracts and processes Heathrow wind data
  - `app/wind_direction_app.py`: Interactive Streamlit application for exploring wind patterns
  - `compare_wind_sources.py`: Aligns and compares data from both sources

- **Jupyter Notebook**:
  - `wind_direction_comparison_analysis.ipynb`: Contains the full technical analysis with visualizations

- **Data Directories**:
  - `/heathrow_data/`: Contains CSV files with Heathrow wind direction data
  - `/openmeteo_data/`: Contains CSV files with Open-Meteo API wind data
  - `/comparison_results/`: Output visualizations and comparison metrics

## Conclusions

The Open-Meteo API provides wind direction data that strongly agrees with official Heathrow measurements, confirming its validity as a data source. When properly adjusted for the systematic bias of approximately +4.1%, it can serve as a reliable substitute for official Heathrow data, particularly for:
- Historical trend analysis
- Pattern identification
- Comparative studies

## Technical Recommendations

1. Apply a correction factor of +4.1% to Open-Meteo westerly percentages when direct comparison with Heathrow official data is required.
2. Consider using season-specific correction factors for highest precision.
3. Include the RMSE of 6.4% as a measure of typical uncertainty when reporting results based on Open-Meteo data.

## Usage

### Running the Analysis Scripts

To run the analysis:

1. Ensure you have Python 3.x installed with required packages (pandas, numpy, matplotlib, seaborn, scipy)
2. Run the data extraction scripts to gather data from both sources
3. Execute the comparison script to generate statistical metrics
4. Open the Jupyter notebook for detailed analysis and visualizations

### Running the Interactive App

#### Local Installation

```bash
# Install required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/wind_direction_app.py
```

#### Using Docker

You can run the application using Docker:

```bash
# Pull the image from GitHub Container Registry
docker pull ghcr.io/emanuelef/historical-wind-direction/wind-direction-app:main

# Run the container
docker run -p 8501:8501 ghcr.io/emanuelef/historical-wind-direction/wind-direction-app:main
```

Or build and run locally:

```bash
# Build the Docker image
docker build -t wind-direction-app .

# Run the container
docker run -p 8501:8501 wind-direction-app
```

Once running, access the application at http://localhost:8501
