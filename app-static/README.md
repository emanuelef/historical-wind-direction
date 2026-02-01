# Climate Data Explorer - Static

A fully static TypeScript frontend for GitHub Pages. No backend required - calls Open-Meteo API directly from the browser.

## Features

- **Wind Direction Analysis** - E/W wind percentage heatmaps, monthly/yearly trends
- **Rainfall Analysis** - Monthly totals, averages, yearly comparisons
- **Temperature Comparison** - Compare two locations side-by-side

## Quick Start

```bash
cd app-static

# Install dependencies
npm install

# Run dev server
npm run dev

# Build for production
npm run build
```

## Deploy to GitHub Pages

### Option 1: GitHub Actions (Recommended)

1. Copy `.github/workflows/deploy-static.yml` to your repo's `.github/workflows/`
2. Enable GitHub Pages in repo Settings > Pages > Source: GitHub Actions
3. Push to main branch - the site will be built and deployed automatically

### Option 2: Manual

```bash
npm run build
# Upload contents of `dist/` to your gh-pages branch or hosting service
```

## Tech Stack

- **TypeScript** - Type-safe JavaScript
- **Vite** - Fast build tool
- **Plotly.js** - Interactive charts (loaded from CDN)
- **Leaflet** - Interactive maps (loaded from CDN)
- **Open-Meteo API** - Free weather data (called directly from browser)

## Configuration

For GitHub Pages deployment, update `vite.config.ts`:

```ts
export default defineConfig({
  base: '/your-repo-name/',  // Change this to your repo name
  // ...
})
```

## Structure

```
app-static/
├── index.html          # Main HTML file
├── src/
│   ├── main.ts         # App logic, UI handling
│   ├── api.ts          # Open-Meteo API client
│   └── styles.css      # Styles
├── package.json        # Dependencies
├── tsconfig.json       # TypeScript config
└── vite.config.ts      # Vite config
```
