import { fetchWindData, fetchRainData, fetchTempData, WindData, RainData, TempData } from './api';

declare const L: typeof import('leaflet');
declare const Plotly: {
  newPlot: (el: string | HTMLElement, data: object[], layout?: object, config?: object) => void;
};

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// State
interface Location {
  lat: number;
  lon: number;
}

const state = {
  wind: { lat: 51.47, lon: -0.4543 } as Location,
  rain: { lat: 51.47, lon: -0.4543 } as Location,
  temp: [] as Location[]
};

let windMap: L.Map;
let rainMap: L.Map;
let tempMap: L.Map;
let windMarker: L.Marker;
let rainMarker: L.Marker;
let tempMarkers: L.Marker[] = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initMaps();
  initButtons();
});

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = (btn as HTMLElement).dataset.tab!;

      // Update buttons
      document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.toggle('active', b === btn);
        b.setAttribute('aria-selected', (b === btn).toString());
      });

      // Update content
      document.querySelectorAll('.tab-content').forEach(c => {
        c.classList.toggle('hidden', c.id !== `${tab}-tab`);
      });

      // Resize map
      setTimeout(() => {
        if (tab === 'wind') windMap?.invalidateSize();
        if (tab === 'rain') rainMap?.invalidateSize();
        if (tab === 'temp') tempMap?.invalidateSize();
      }, 100);
    });
  });
}

function initMaps() {
  const tileLayer = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
  const attribution = '&copy; OpenStreetMap, &copy; CARTO';

  // Wind map
  windMap = L.map('wind-map').setView([state.wind.lat, state.wind.lon], 4);
  L.tileLayer(tileLayer, { attribution }).addTo(windMap);
  windMarker = L.marker([state.wind.lat, state.wind.lon]).addTo(windMap);

  windMap.on('click', (e: L.LeafletMouseEvent) => {
    state.wind.lat = e.latlng.lat;
    state.wind.lon = e.latlng.lng;
    windMarker.setLatLng(e.latlng);
    updateCoords('wind');
  });

  // Rain map
  rainMap = L.map('rain-map').setView([state.rain.lat, state.rain.lon], 4);
  L.tileLayer(tileLayer, { attribution }).addTo(rainMap);
  rainMarker = L.marker([state.rain.lat, state.rain.lon]).addTo(rainMap);

  rainMap.on('click', (e: L.LeafletMouseEvent) => {
    state.rain.lat = e.latlng.lat;
    state.rain.lon = e.latlng.lng;
    rainMarker.setLatLng(e.latlng);
    updateCoords('rain');
  });

  // Temp map
  tempMap = L.map('temp-map').setView([45.46, 9.19], 4);
  L.tileLayer(tileLayer, { attribution }).addTo(tempMap);

  tempMap.on('click', (e: L.LeafletMouseEvent) => {
    if (state.temp.length >= 2) {
      state.temp.shift();
      tempMarkers[0]?.remove();
      tempMarkers.shift();
    }

    state.temp.push({ lat: e.latlng.lat, lon: e.latlng.lng });

    const color = state.temp.length === 1 ? '#4285F4' : '#ea4335';
    const marker = L.marker(e.latlng, {
      icon: L.divIcon({
        className: 'custom-marker',
        html: `<div style="background:${color};width:16px;height:16px;border-radius:50%;border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.3);"></div>`,
        iconSize: [16, 16],
        iconAnchor: [8, 8]
      })
    }).addTo(tempMap);
    tempMarkers.push(marker);

    updateTempLocations();
  });
}

function initButtons() {
  // Wind
  document.getElementById('wind-reset')?.addEventListener('click', () => {
    state.wind = { lat: 51.47, lon: -0.4543 };
    windMarker.setLatLng([state.wind.lat, state.wind.lon]);
    windMap.setView([state.wind.lat, state.wind.lon], 4);
    updateCoords('wind');
  });

  document.getElementById('wind-analyze')?.addEventListener('click', analyzeWind);

  // Rain
  document.getElementById('rain-reset')?.addEventListener('click', () => {
    state.rain = { lat: 51.47, lon: -0.4543 };
    rainMarker.setLatLng([state.rain.lat, state.rain.lon]);
    rainMap.setView([state.rain.lat, state.rain.lon], 4);
    updateCoords('rain');
  });

  document.getElementById('rain-analyze')?.addEventListener('click', analyzeRain);

  // Temp
  document.getElementById('temp-clear')?.addEventListener('click', () => {
    state.temp = [];
    tempMarkers.forEach(m => m.remove());
    tempMarkers = [];
    updateTempLocations();
    document.getElementById('temp-results')?.classList.add('hidden');
  });

  document.getElementById('temp-compare')?.addEventListener('click', compareTemp);
}

function updateCoords(type: 'wind' | 'rain') {
  const loc = state[type];
  const el = document.getElementById(`${type}-coords`);
  if (el) {
    el.textContent = `${loc.lat.toFixed(4)}, ${loc.lon.toFixed(4)}`;
  }
}

function updateTempLocations() {
  const container = document.getElementById('temp-locations');
  const btn = document.getElementById('temp-compare') as HTMLButtonElement;

  if (container) {
    if (state.temp.length === 0) {
      container.innerHTML = '<span class="location-tag">No locations selected</span>';
    } else {
      container.innerHTML = state.temp.map((loc, i) =>
        `<span class="location-tag loc${i + 1}">Loc ${i + 1}: ${loc.lat.toFixed(4)}, ${loc.lon.toFixed(4)}</span>`
      ).join('');
    }
  }

  if (btn) {
    btn.disabled = state.temp.length !== 2;
  }
}

// Analysis functions
async function analyzeWind() {
  const loading = document.getElementById('wind-loading')!;
  const results = document.getElementById('wind-results')!;

  loading.classList.remove('hidden');
  results.classList.add('hidden');

  try {
    const data = await fetchWindData(state.wind.lat, state.wind.lon);
    renderWindResults(data);
    results.classList.remove('hidden');
  } catch (err) {
    alert('Error fetching wind data: ' + (err as Error).message);
  } finally {
    loading.classList.add('hidden');
  }
}

async function analyzeRain() {
  const loading = document.getElementById('rain-loading')!;
  const results = document.getElementById('rain-results')!;

  loading.classList.remove('hidden');
  results.classList.add('hidden');

  try {
    const data = await fetchRainData(state.rain.lat, state.rain.lon);
    renderRainResults(data);
    results.classList.remove('hidden');
  } catch (err) {
    alert('Error fetching rain data: ' + (err as Error).message);
  } finally {
    loading.classList.add('hidden');
  }
}

async function compareTemp() {
  if (state.temp.length !== 2) return;

  const loading = document.getElementById('temp-loading')!;
  const results = document.getElementById('temp-results')!;

  loading.classList.remove('hidden');
  results.classList.add('hidden');

  try {
    const [data1, data2] = await Promise.all([
      fetchTempData(state.temp[0].lat, state.temp[0].lon),
      fetchTempData(state.temp[1].lat, state.temp[1].lon)
    ]);
    renderTempResults(data1, data2);
    results.classList.remove('hidden');
  } catch (err) {
    alert('Error fetching temperature data: ' + (err as Error).message);
  } finally {
    loading.classList.add('hidden');
  }
}

// Render functions
function renderWindResults(data: WindData) {
  const results = document.getElementById('wind-results')!;

  results.innerHTML = `
    <div class="card">
      <h3>Key Statistics</h3>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Avg. Westerly</div>
          <div class="metric-value">${data.stats.avgWesterly}%</div>
        </div>
        <div class="metric">
          <div class="metric-label">Predominant Direction</div>
          <div class="metric-value">${data.stats.predominant}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Windiest W Month</div>
          <div class="metric-value">${MONTHS[data.stats.maxWMonth - 1]}</div>
          <div class="metric-delta">${data.stats.maxWPct}% W</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="sub-tabs" id="wind-sub-tabs">
        <button class="sub-tab-btn active" data-subtab="wind-heatmap">Westerly %</button>
        <button class="sub-tab-btn" data-subtab="wind-monthly">Monthly Stats</button>
        <button class="sub-tab-btn" data-subtab="wind-yearly">Yearly Trend</button>
      </div>
      <div id="wind-heatmap" class="sub-content"><div id="wind-heatmap-chart" class="chart-container"></div></div>
      <div id="wind-monthly" class="sub-content hidden"><div id="wind-monthly-chart" class="chart-container"></div></div>
      <div id="wind-yearly" class="sub-content hidden"><div id="wind-yearly-chart" class="chart-container"></div></div>
    </div>
  `;

  // Setup sub-tabs
  setupSubTabs('wind-sub-tabs');

  // Custom RdYlGn colorscale matching seaborn
  const windColorscale: [number, string][] = [
    [0, '#a50026'],     // dark red
    [0.1, '#d73027'],   // red
    [0.2, '#f46d43'],   // orange-red
    [0.3, '#fdae61'],   // orange
    [0.4, '#fee08b'],   // yellow-orange
    [0.5, '#ffffbf'],   // yellow
    [0.6, '#d9ef8b'],   // yellow-green
    [0.7, '#a6d96a'],   // light green
    [0.8, '#66bd63'],   // green
    [0.9, '#1a9850'],   // dark green
    [1, '#006837']      // darkest green
  ];

  // Replace 0 values with null (no data) so they appear blank
  const windValuesBase = data.heatmap.values.map(row =>
    row.map(v => v === 0 ? null : v)
  );

  // Calculate yearly averages (average of each column)
  const yearlyAvg = data.heatmap.years.map((_, yearIdx) => {
    const colValues = data.heatmap.values
      .map(row => row[yearIdx])
      .filter(v => v !== 0 && v !== null && v !== undefined);
    if (colValues.length === 0) return null;
    return colValues.reduce((a, b) => a + b, 0) / colValues.length;
  });

  // Add average row at the bottom
  const windValues = [...windValuesBase, yearlyAvg];
  const windYLabels = [...data.heatmap.months.map(m => MONTHS[m - 1]), 'Average'];

  const windTextBase = data.heatmap.values.map(row =>
    row.map(v => v === 0 ? '' : v.toFixed(1))
  );
  const avgText = yearlyAvg.map(v => v === null ? '' : v.toFixed(1));
  const windText = [...windTextBase, avgText];

  // Heatmap
  Plotly.newPlot('wind-heatmap-chart', [{
    z: windValues,
    x: data.heatmap.years,
    y: windYLabels,
    type: 'heatmap',
    colorscale: windColorscale,
    zmin: 0,
    zmax: 100,
    xgap: 1,
    ygap: 1,
    text: windText,
    texttemplate: '%{text}',
    textfont: { size: 12, color: '#000' },
    hoverinfo: 'z',
    colorbar: { title: '% Westerly Winds', len: 0.9 }
  }], {
    title: 'Westerly Wind Percentage per Month per Year',
    height: 700,
    width: 1000,
    margin: { t: 50, l: 70, r: 120, b: 50 },
    yaxis: { autorange: 'reversed' }
  }, { responsive: true });

  // Monthly
  Plotly.newPlot('wind-monthly-chart', [
    { x: MONTHS, y: data.monthlyAvg.E, name: 'Easterly', type: 'bar', marker: { color: '#ea4335' } },
    { x: MONTHS, y: data.monthlyAvg.W, name: 'Westerly', type: 'bar', marker: { color: '#34a853' } }
  ], {
    title: 'Average E/W Wind by Month',
    barmode: 'group',
    yaxis: { title: '%' }
  }, { responsive: true });

  // Yearly
  Plotly.newPlot('wind-yearly-chart', [{
    x: data.yearlyAvg.years,
    y: data.yearlyAvg.W,
    type: 'bar',
    marker: { color: data.yearlyAvg.W.map(v => v > 50 ? '#34a853' : '#ea4335') }
  }, {
    x: data.yearlyAvg.years,
    y: Array(data.yearlyAvg.years.length).fill(50),
    type: 'scatter',
    mode: 'lines',
    name: '50%',
    line: { dash: 'dash', color: '#666' }
  }], {
    title: 'Yearly Westerly %',
    yaxis: { range: [0, 100] }
  }, { responsive: true });
}

function renderRainResults(data: RainData) {
  const results = document.getElementById('rain-results')!;

  results.innerHTML = `
    <div class="card">
      <h3>Key Statistics</h3>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Avg. Yearly Rainfall</div>
          <div class="metric-value">${data.stats.avgYearly} mm</div>
        </div>
        <div class="metric">
          <div class="metric-label">Rainy Days</div>
          <div class="metric-value">${data.stats.rainyPct}%</div>
          <div class="metric-delta">${data.stats.rainyDays} of ${data.stats.totalDays} days</div>
        </div>
        <div class="metric">
          <div class="metric-label">Max Daily</div>
          <div class="metric-value">${data.stats.maxDaily} mm</div>
          <div class="metric-delta">${data.stats.maxDailyDate}</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="sub-tabs" id="rain-sub-tabs">
        <button class="sub-tab-btn active" data-subtab="rain-heatmap">Monthly Totals</button>
        <button class="sub-tab-btn" data-subtab="rain-monthly">Monthly Avg</button>
        <button class="sub-tab-btn" data-subtab="rain-yearly">Yearly Totals</button>
      </div>
      <div id="rain-heatmap" class="sub-content"><div id="rain-heatmap-chart" class="chart-container"></div></div>
      <div id="rain-monthly" class="sub-content hidden"><div id="rain-monthly-chart" class="chart-container"></div></div>
      <div id="rain-yearly" class="sub-content hidden"><div id="rain-yearly-chart" class="chart-container"></div></div>
    </div>
  `;

  setupSubTabs('rain-sub-tabs');

  // Replace 0 values with null (no data) so they appear blank
  const rainValuesBase = data.heatmap.values.map(row =>
    row.map(v => v === 0 ? null : v)
  );

  // Calculate yearly averages (average of each column)
  const rainYearlyAvg = data.heatmap.years.map((_, yearIdx) => {
    const colValues = data.heatmap.values
      .map(row => row[yearIdx])
      .filter(v => v !== 0 && v !== null && v !== undefined);
    if (colValues.length === 0) return null;
    return colValues.reduce((a, b) => a + b, 0) / colValues.length;
  });

  // Add average row at the bottom
  const rainValues = [...rainValuesBase, rainYearlyAvg];
  const rainYLabels = [...data.heatmap.months.map(m => MONTHS[m - 1]), 'Average'];

  const rainTextBase = data.heatmap.values.map(row =>
    row.map(v => v === 0 ? '' : v.toFixed(0))
  );
  const rainAvgText = rainYearlyAvg.map(v => v === null ? '' : v.toFixed(0));
  const rainText = [...rainTextBase, rainAvgText];

  // Heatmap
  Plotly.newPlot('rain-heatmap-chart', [{
    z: rainValues,
    x: data.heatmap.years,
    y: rainYLabels,
    type: 'heatmap',
    colorscale: 'Blues',
    reversescale: true,
    xgap: 1,
    ygap: 1,
    text: rainText,
    texttemplate: '%{text}',
    textfont: { size: 11 },
    hoverinfo: 'z',
    colorbar: { title: 'mm' }
  }], {
    title: 'Monthly Rainfall (mm)',
    height: 600,
    width: 1000,
    margin: { t: 40, l: 60, r: 120, b: 50 },
    yaxis: { autorange: 'reversed' }
  }, { responsive: true });

  // Monthly
  Plotly.newPlot('rain-monthly-chart', [{
    x: MONTHS,
    y: data.monthlyAvg,
    type: 'bar',
    marker: { color: '#4285F4' }
  }], {
    title: 'Average Monthly Rainfall',
    yaxis: { title: 'mm' }
  }, { responsive: true });

  // Yearly
  const avgYearly = data.yearlyTotal.totals.reduce((a, b) => a + b, 0) / data.yearlyTotal.totals.length;
  Plotly.newPlot('rain-yearly-chart', [{
    x: data.yearlyTotal.years,
    y: data.yearlyTotal.totals,
    type: 'bar',
    marker: { color: '#4285F4' }
  }, {
    x: data.yearlyTotal.years,
    y: Array(data.yearlyTotal.years.length).fill(avgYearly),
    type: 'scatter',
    mode: 'lines',
    name: `Avg: ${avgYearly.toFixed(0)} mm`,
    line: { dash: 'dash', color: '#ea4335' }
  }], {
    title: 'Yearly Total Rainfall',
    yaxis: { title: 'mm' }
  }, { responsive: true });
}

function renderTempResults(data1: TempData, data2: TempData) {
  const results = document.getElementById('temp-results')!;
  const loc1 = state.temp[0];
  const loc2 = state.temp[1];
  const diff = data2.stats.maxTemp - data1.stats.maxTemp;

  results.innerHTML = `
    <div class="card">
      <h3>Key Comparison</h3>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Max Temp (Loc 1)</div>
          <div class="metric-value">${data1.stats.maxTemp}°C</div>
        </div>
        <div class="metric">
          <div class="metric-label">Max Temp (Loc 2)</div>
          <div class="metric-value">${data2.stats.maxTemp}°C</div>
          <div class="metric-delta">${diff > 0 ? '+' : ''}${diff.toFixed(1)}°C</div>
        </div>
        <div class="metric">
          <div class="metric-label">Hottest Month (Loc 1)</div>
          <div class="metric-value">${MONTHS[data1.stats.hottestMonth - 1]}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Hottest Month (Loc 2)</div>
          <div class="metric-value">${MONTHS[data2.stats.hottestMonth - 1]}</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="grid-2">
        <div>
          <h4>Location 1: ${loc1.lat.toFixed(4)}, ${loc1.lon.toFixed(4)}</h4>
          <div id="temp-chart1" class="chart-container"></div>
        </div>
        <div>
          <h4>Location 2: ${loc2.lat.toFixed(4)}, ${loc2.lon.toFixed(4)}</h4>
          <div id="temp-chart2" class="chart-container"></div>
        </div>
      </div>
    </div>

    <div class="card">
      <h4>Temperature Difference (Loc 2 - Loc 1)</h4>
      <div id="temp-diff-chart" class="chart-container"></div>
    </div>
  `;

  const tempColorscale: [number, string][] = [
    [0, '#ADD8E6'],
    [0.4, '#FFFF66'],
    [0.7, '#FFA500'],
    [1, '#FF4500']
  ];

  // Replace 0 values with null (no data) so they appear blank
  const temp1ValuesBase = data1.heatmap.values.map(row =>
    row.map(v => v === 0 ? null : v)
  );
  const temp2ValuesBase = data2.heatmap.values.map(row =>
    row.map(v => v === 0 ? null : v)
  );

  // Calculate yearly averages for both locations
  const temp1YearlyAvg = data1.heatmap.years.map((_, yearIdx) => {
    const colValues = data1.heatmap.values
      .map(row => row[yearIdx])
      .filter(v => v !== 0 && v !== null && v !== undefined);
    if (colValues.length === 0) return null;
    return colValues.reduce((a, b) => a + b, 0) / colValues.length;
  });
  const temp2YearlyAvg = data2.heatmap.years.map((_, yearIdx) => {
    const colValues = data2.heatmap.values
      .map(row => row[yearIdx])
      .filter(v => v !== 0 && v !== null && v !== undefined);
    if (colValues.length === 0) return null;
    return colValues.reduce((a, b) => a + b, 0) / colValues.length;
  });

  // Add average row at the bottom
  const temp1Values = [...temp1ValuesBase, temp1YearlyAvg];
  const temp2Values = [...temp2ValuesBase, temp2YearlyAvg];
  const tempYLabels = [...data1.heatmap.months.map(m => MONTHS[m - 1]), 'Average'];

  const temp1TextBase = data1.heatmap.values.map(row =>
    row.map(v => v === 0 ? '' : v.toFixed(1))
  );
  const temp1AvgText = temp1YearlyAvg.map(v => v === null ? '' : v.toFixed(1));
  const temp1Text = [...temp1TextBase, temp1AvgText];

  const temp2TextBase = data2.heatmap.values.map(row =>
    row.map(v => v === 0 ? '' : v.toFixed(1))
  );
  const temp2AvgText = temp2YearlyAvg.map(v => v === null ? '' : v.toFixed(1));
  const temp2Text = [...temp2TextBase, temp2AvgText];

  // Heatmap 1
  Plotly.newPlot('temp-chart1', [{
    z: temp1Values,
    x: data1.heatmap.years,
    y: tempYLabels,
    type: 'heatmap',
    colorscale: tempColorscale,
    zmin: 0,
    zmax: 45,
    xgap: 1,
    ygap: 1,
    text: temp1Text,
    texttemplate: '%{text}',
    textfont: { size: 11 },
    hoverinfo: 'z',
    colorbar: { title: '°C' }
  }], { height: 550, margin: { t: 20, l: 60, r: 100 }, yaxis: { autorange: 'reversed' } }, { responsive: true });

  // Heatmap 2
  Plotly.newPlot('temp-chart2', [{
    z: temp2Values,
    x: data2.heatmap.years,
    y: tempYLabels,
    type: 'heatmap',
    colorscale: tempColorscale,
    zmin: 0,
    zmax: 45,
    xgap: 1,
    ygap: 1,
    text: temp2Text,
    texttemplate: '%{text}',
    textfont: { size: 11 },
    hoverinfo: 'z',
    colorbar: { title: '°C' }
  }], { height: 550, margin: { t: 20, l: 60, r: 100 }, yaxis: { autorange: 'reversed' } }, { responsive: true });

  // Diff heatmap - handle null values
  const diffValuesBase = data2.heatmap.values.map((row, i) =>
    row.map((val, j) => {
      const v1 = data1.heatmap.values[i]?.[j] || 0;
      if (val === 0 || v1 === 0) return null;
      return val - v1;
    })
  );
  // Calculate average diff for each year
  const diffYearlyAvg = data2.heatmap.years.map((_, yearIdx) => {
    const t1Avg = temp1YearlyAvg[yearIdx];
    const t2Avg = temp2YearlyAvg[yearIdx];
    if (t1Avg === null || t2Avg === null) return null;
    return t2Avg - t1Avg;
  });
  const diffValues = [...diffValuesBase, diffYearlyAvg];

  const diffTextBase = diffValuesBase.map(row =>
    row.map(v => v === null ? '' : v.toFixed(1))
  );
  const diffAvgText = diffYearlyAvg.map(v => v === null ? '' : v.toFixed(1));
  const diffText = [...diffTextBase, diffAvgText];

  Plotly.newPlot('temp-diff-chart', [{
    z: diffValues,
    x: data2.heatmap.years,
    y: tempYLabels,
    type: 'heatmap',
    colorscale: 'RdBu',
    zmid: 0,
    zmin: -20,
    zmax: 20,
    xgap: 1,
    ygap: 1,
    text: diffText,
    texttemplate: '%{text}',
    textfont: { size: 11 },
    hoverinfo: 'z',
    colorbar: { title: '°C' }
  }], { height: 550, width: 1000, margin: { t: 20, l: 60, r: 100 }, yaxis: { autorange: 'reversed' } }, { responsive: true });
}

function setupSubTabs(containerId: string) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.querySelectorAll('.sub-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const subtab = (btn as HTMLElement).dataset.subtab!;
      const card = container.closest('.card')!;

      // Update buttons
      container.querySelectorAll('.sub-tab-btn').forEach(b => {
        b.classList.toggle('active', b === btn);
      });

      // Update content
      card.querySelectorAll('.sub-content').forEach(c => {
        c.classList.toggle('hidden', c.id !== subtab);
      });

      // Trigger resize for Plotly
      window.dispatchEvent(new Event('resize'));
    });
  });
}
