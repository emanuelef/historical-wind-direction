# Fast Climate Data Explorer

A lightning-fast alternative to the Streamlit version, built with FastAPI + vanilla JavaScript.

## Why is it faster?

1. **Async data fetching** - Uses httpx async client for non-blocking API calls
2. **Client-side rendering** - Charts render in the browser with Plotly.js
3. **No full page re-renders** - Unlike Streamlit, only updated components change
4. **In-memory caching** - API responses cached for 1 hour
5. **Parallel requests** - Temperature comparison fetches both locations simultaneously

## Quick Start

```bash
# Navigate to this directory
cd app-fast

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and run
uv sync
uv run uvicorn main:app --port 8000

# Or with auto-reload for development
uv run uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000 in your browser.

### Alternative (pip)

```bash
pip install -r requirements.txt
python main.py
```

## Features

- **Wind Direction Analysis** - Heatmaps, monthly/yearly stats, streak analysis
- **Rainfall Analysis** - Monthly totals, daily patterns, dry/wet periods
- **Temperature Comparison** - Compare two locations side-by-side

## API Endpoints

- `GET /api/wind/{lat}/{lon}` - Wind direction data
- `GET /api/rain/{lat}/{lon}` - Rainfall data
- `GET /api/temperature/{lat}/{lon}` - Temperature data

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla JavaScript
- **Charts**: Plotly.js
- **Maps**: Leaflet.js
- **Data Source**: Open-Meteo Archive API
