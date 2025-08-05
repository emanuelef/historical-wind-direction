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

```bash
# Pull the latest image
docker pull ghcr.io/emanuelef/historical-wind-direction/wind-direction-app:main

# Run the container
docker run -p 8501:8501 ghcr.io/emanuelef/historical-wind-direction/wind-direction-app:main

# The image supports both amd64 and arm64 architectures (including Apple Silicon M1/M2/M3)
```

Or build and run locally:

```bash
# Build the Docker image
docker build -t climate-data-app .

# Run the container
docker run -p 8501:8501 climate-data-app
```

### Running Locally

```bash
# Install requirements
pip install -r requirements.txt

# Run the unified app
streamlit run app/unified_app.py

# Alternatively, you can run the individual apps
streamlit run app/wind_direction_app.py  # Wind Direction Analysis only
streamlit run app/app.py                 # Temperature Comparison only
```

Once running, access the application at http://localhost:8501

## Data Sources

The application uses historical weather data from:
1. Open-Meteo API historical weather data archive

## Interactive Applications

### Unified App (New!)
The `unified_app.py` combines both the wind direction analysis and temperature comparison tools into a single application with a tab-based interface.

### Wind Direction Analysis
The `wind_direction_app.py` allows users to:
- Select any global location
- Analyze historical wind patterns (east vs west)
- View monthly and yearly trends
- Identify periods of consistent wind patterns

### Temperature Comparison
The `app.py` enables users to:
- Select two different global locations
- Compare apparent temperatures across years
- View the temperature differences between locations

## Technical Details

The repository includes:

- **Application Files**:
  - `app/unified_app.py`: Combined application with both tools
  - `app/wind_direction_app.py`: Wind direction analysis tool
  - `app/app.py`: Temperature comparison tool

- **Docker Configuration**:
  - `Dockerfile`: Container definition for running the app
  - `.github/workflows/docker-publish.yml`: CI/CD pipeline for multi-architecture builds

## Requirements

- Python 3.8+
- Streamlit 1.30.0+
- Pandas, Numpy, Matplotlib
- Folium and Streamlit-Folium for interactive maps
- Seaborn for enhanced visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
