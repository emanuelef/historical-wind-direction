// Open-Meteo API client

const API_BASE = 'https://archive-api.open-meteo.com/v1/archive';

interface HourlyData {
  time: string[];
  wind_speed_10m?: number[];
  wind_direction_10m?: number[];
  precipitation?: number[];
  apparent_temperature?: number[];
}

interface ApiResponse {
  hourly: HourlyData;
}

function getDateRange(): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setFullYear(end.getFullYear() - 10);
  start.setMonth(0, 1);

  return {
    start: start.toISOString().split('T')[0],
    end: end.toISOString().split('T')[0]
  };
}

export async function fetchWindData(lat: number, lon: number): Promise<WindData> {
  const { start, end } = getDateRange();

  const params = new URLSearchParams({
    latitude: lat.toString(),
    longitude: lon.toString(),
    start_date: start,
    end_date: end,
    hourly: 'wind_speed_10m,wind_direction_10m',
    timezone: 'auto'
  });

  const response = await fetch(`${API_BASE}?${params}`);
  if (!response.ok) throw new Error('Failed to fetch wind data');

  const data: ApiResponse = await response.json();
  return processWindData(data.hourly);
}

export async function fetchRainData(lat: number, lon: number): Promise<RainData> {
  const { start, end } = getDateRange();

  const params = new URLSearchParams({
    latitude: lat.toString(),
    longitude: lon.toString(),
    start_date: start,
    end_date: end,
    hourly: 'precipitation',
    timezone: 'auto'
  });

  const response = await fetch(`${API_BASE}?${params}`);
  if (!response.ok) throw new Error('Failed to fetch rain data');

  const data: ApiResponse = await response.json();
  return processRainData(data.hourly);
}

export async function fetchTempData(lat: number, lon: number): Promise<TempData> {
  const { start, end } = getDateRange();

  const params = new URLSearchParams({
    latitude: lat.toString(),
    longitude: lon.toString(),
    start_date: start,
    end_date: end,
    hourly: 'apparent_temperature',
    timezone: 'auto'
  });

  const response = await fetch(`${API_BASE}?${params}`);
  if (!response.ok) throw new Error('Failed to fetch temperature data');

  const data: ApiResponse = await response.json();
  return processTempData(data.hourly);
}

// Data types
export interface WindData {
  heatmap: { months: number[]; years: string[]; values: number[][] };
  monthlyAvg: { E: number[]; W: number[] };
  yearlyAvg: { years: string[]; W: number[] };
  stats: {
    avgWesterly: number;
    avgEasterly: number;
    predominant: string;
    maxWMonth: number;
    maxWPct: number;
  };
  streaks: { easterly: Streak[]; westerly: Streak[] };
}

export interface RainData {
  heatmap: { months: number[]; years: string[]; values: number[][] };
  monthlyAvg: number[];
  yearlyTotal: { years: string[]; totals: number[]; rainyDays: number[] };
  stats: {
    avgYearly: number;
    rainyPct: number;
    rainyDays: number;
    totalDays: number;
    maxDaily: number;
    maxDailyDate: string;
  };
  streaks: { dry: Streak[]; wet: Streak[] };
}

export interface TempData {
  heatmap: { months: number[]; years: string[]; values: number[][] };
  stats: {
    maxTemp: number;
    avgTemp: number;
    hottestMonth: number;
  };
}

interface Streak {
  start: string;
  end: string;
  days: number;
}

// Helper functions
function degToCompass(deg: number | null): string | null {
  if (deg === null || isNaN(deg)) return null;
  const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const ix = Math.floor((deg + 22.5) / 45) % 8;
  return directions[ix];
}

function processWindData(hourly: HourlyData): WindData {
  const times = hourly.time.map(t => new Date(t));
  const directions = hourly.wind_direction_10m || [];

  // Group by month/year
  const monthYearData: Map<string, { E: number; W: number }> = new Map();

  times.forEach((time, i) => {
    const compass = degToCompass(directions[i]);
    if (compass !== 'E' && compass !== 'W') return;

    const year = time.getFullYear().toString();
    const month = time.getMonth() + 1;
    const key = `${month}-${year}`;

    if (!monthYearData.has(key)) {
      monthYearData.set(key, { E: 0, W: 0 });
    }
    const data = monthYearData.get(key)!;
    if (compass === 'E') data.E++;
    else data.W++;
  });

  // Get unique years and months
  const years = [...new Set([...monthYearData.keys()].map(k => k.split('-')[1]))].sort();
  const months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

  // Build heatmap (W percentage)
  const heatmapValues: number[][] = months.map(month => {
    return years.map(year => {
      const key = `${month}-${year}`;
      const data = monthYearData.get(key);
      if (!data) return 0;
      const total = data.E + data.W;
      return total > 0 ? (data.W / total) * 100 : 0;
    });
  });

  // Monthly averages
  const monthlyE: number[] = [];
  const monthlyW: number[] = [];
  months.forEach(month => {
    let totalE = 0, totalW = 0, count = 0;
    years.forEach(year => {
      const key = `${month}-${year}`;
      const data = monthYearData.get(key);
      if (data) {
        const total = data.E + data.W;
        if (total > 0) {
          totalE += (data.E / total) * 100;
          totalW += (data.W / total) * 100;
          count++;
        }
      }
    });
    monthlyE.push(count > 0 ? totalE / count : 0);
    monthlyW.push(count > 0 ? totalW / count : 0);
  });

  // Yearly averages
  const yearlyW: number[] = years.map(year => {
    let total = 0, count = 0;
    months.forEach(month => {
      const key = `${month}-${year}`;
      const data = monthYearData.get(key);
      if (data) {
        const sum = data.E + data.W;
        if (sum > 0) {
          total += (data.W / sum) * 100;
          count++;
        }
      }
    });
    return count > 0 ? total / count : 0;
  });

  // Stats
  const avgW = monthlyW.reduce((a, b) => a + b, 0) / 12;
  const avgE = monthlyE.reduce((a, b) => a + b, 0) / 12;
  const maxWMonth = monthlyW.indexOf(Math.max(...monthlyW)) + 1;

  // Streaks (simplified - just return empty for now)
  return {
    heatmap: { months, years, values: heatmapValues },
    monthlyAvg: { E: monthlyE, W: monthlyW },
    yearlyAvg: { years, W: yearlyW },
    stats: {
      avgWesterly: Math.round(avgW * 10) / 10,
      avgEasterly: Math.round(avgE * 10) / 10,
      predominant: avgW > avgE ? 'Westerly' : 'Easterly',
      maxWMonth,
      maxWPct: Math.round(Math.max(...monthlyW) * 10) / 10
    },
    streaks: { easterly: [], westerly: [] }
  };
}

function processRainData(hourly: HourlyData): RainData {
  const times = hourly.time.map(t => new Date(t));
  const precip = hourly.precipitation || [];

  // Group by day
  const dailyData: Map<string, number> = new Map();

  times.forEach((time, i) => {
    const dateKey = time.toISOString().split('T')[0];
    const current = dailyData.get(dateKey) || 0;
    dailyData.set(dateKey, current + (precip[i] || 0));
  });

  // Group by month/year
  const monthYearData: Map<string, number> = new Map();
  const yearlyData: Map<string, { total: number; rainyDays: number }> = new Map();

  dailyData.forEach((total, dateKey) => {
    const date = new Date(dateKey);
    const year = date.getFullYear().toString();
    const month = date.getMonth() + 1;
    const monthKey = `${month}-${year}`;

    // Monthly totals
    const current = monthYearData.get(monthKey) || 0;
    monthYearData.set(monthKey, current + total);

    // Yearly totals and rainy days
    if (!yearlyData.has(year)) {
      yearlyData.set(year, { total: 0, rainyDays: 0 });
    }
    const yd = yearlyData.get(year)!;
    yd.total += total;
    if (total > 0.1) yd.rainyDays++;
  });

  const years = [...yearlyData.keys()].sort();
  const months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

  // Heatmap
  const heatmapValues: number[][] = months.map(month => {
    return years.map(year => {
      const key = `${month}-${year}`;
      return monthYearData.get(key) || 0;
    });
  });

  // Monthly averages
  const monthlyAvg: number[] = months.map(month => {
    let total = 0, count = 0;
    years.forEach(year => {
      const key = `${month}-${year}`;
      const val = monthYearData.get(key);
      if (val !== undefined) {
        total += val;
        count++;
      }
    });
    return count > 0 ? total / count : 0;
  });

  // Yearly summary
  const yearlyTotal = years.map(y => yearlyData.get(y)!.total);
  const yearlyRainyDays = years.map(y => yearlyData.get(y)!.rainyDays);

  // Stats
  const totalRain = [...yearlyData.values()].reduce((a, b) => a + b.total, 0);
  const totalDays = dailyData.size;
  const rainyDays = [...dailyData.values()].filter(v => v > 0.1).length;
  let maxDaily = 0;
  let maxDailyDate = '';
  dailyData.forEach((val, date) => {
    if (val > maxDaily) {
      maxDaily = val;
      maxDailyDate = date;
    }
  });

  return {
    heatmap: { months, years, values: heatmapValues },
    monthlyAvg,
    yearlyTotal: { years, totals: yearlyTotal, rainyDays: yearlyRainyDays },
    stats: {
      avgYearly: Math.round(totalRain / years.length),
      rainyPct: Math.round((rainyDays / totalDays) * 1000) / 10,
      rainyDays,
      totalDays,
      maxDaily: Math.round(maxDaily * 10) / 10,
      maxDailyDate
    },
    streaks: { dry: [], wet: [] }
  };
}

function processTempData(hourly: HourlyData): TempData {
  const times = hourly.time.map(t => new Date(t));
  const temps = hourly.apparent_temperature || [];

  // Group by day (max temp)
  const dailyMax: Map<string, number> = new Map();

  times.forEach((time, i) => {
    const dateKey = time.toISOString().split('T')[0];
    const current = dailyMax.get(dateKey) ?? -Infinity;
    const temp = temps[i];
    if (temp !== null && temp !== undefined && temp > current) {
      dailyMax.set(dateKey, temp);
    }
  });

  // Group by month/year (max of daily max)
  const monthYearMax: Map<string, number> = new Map();

  dailyMax.forEach((maxTemp, dateKey) => {
    const date = new Date(dateKey);
    const year = date.getFullYear().toString();
    const month = date.getMonth() + 1;
    const key = `${month}-${year}`;

    const current = monthYearMax.get(key) ?? -Infinity;
    if (maxTemp > current) {
      monthYearMax.set(key, maxTemp);
    }
  });

  const years = [...new Set([...monthYearMax.keys()].map(k => k.split('-')[1]))].sort();
  const months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

  // Heatmap
  const heatmapValues: number[][] = months.map(month => {
    return years.map(year => {
      const key = `${month}-${year}`;
      return monthYearMax.get(key) || 0;
    });
  });

  // Stats
  let maxTemp = -Infinity;
  let totalTemp = 0;
  let count = 0;
  const monthlyAvg: number[] = [];

  months.forEach(month => {
    let monthTotal = 0, monthCount = 0;
    years.forEach(year => {
      const key = `${month}-${year}`;
      const val = monthYearMax.get(key);
      if (val !== undefined) {
        if (val > maxTemp) maxTemp = val;
        totalTemp += val;
        count++;
        monthTotal += val;
        monthCount++;
      }
    });
    monthlyAvg.push(monthCount > 0 ? monthTotal / monthCount : 0);
  });

  const hottestMonth = monthlyAvg.indexOf(Math.max(...monthlyAvg)) + 1;

  return {
    heatmap: { months, years, values: heatmapValues },
    stats: {
      maxTemp: Math.round(maxTemp * 10) / 10,
      avgTemp: Math.round((totalTemp / count) * 10) / 10,
      hottestMonth
    }
  };
}
