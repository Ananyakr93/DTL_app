import type { AQIColorClass, HourlyPrediction, Scenario, ScenarioOption, HistoricalDataPoint } from './types';

// AQI Helper Functions
export function getAQIClass(aqi: number): AQIColorClass {
    if (aqi <= 50) return 'good';
    if (aqi <= 100) return 'satisfactory';
    if (aqi <= 200) return 'moderate';
    if (aqi <= 300) return 'poor';
    if (aqi <= 400) return 'very-poor';
    return 'severe';
}

export function getAQIStatus(aqi: number): string {
    if (aqi <= 50) return 'Good';
    if (aqi <= 100) return 'Satisfactory';
    if (aqi <= 200) return 'Moderate';
    if (aqi <= 300) return 'Poor';
    if (aqi <= 400) return 'Very Poor';
    return 'Severe';
}

export function getAQIColor(colorClass: AQIColorClass): string {
    const colors: Record<AQIColorClass, string> = {
        good: '#22c55e',
        satisfactory: '#84cc16',
        moderate: '#eab308',
        poor: '#f97316',
        'very-poor': '#ef4444',
        severe: '#9333ea',
    };
    return colors[colorClass];
}

export function getAQIBgClass(colorClass: AQIColorClass): string {
    const bgClasses: Record<AQIColorClass, string> = {
        good: 'bg-green-500',
        satisfactory: 'bg-lime-500',
        moderate: 'bg-yellow-500',
        poor: 'bg-orange-500',
        'very-poor': 'bg-red-500',
        severe: 'bg-purple-600',
    };
    return bgClasses[colorClass];
}

export function getAQITextClass(colorClass: AQIColorClass): string {
    const textClasses: Record<AQIColorClass, string> = {
        good: 'text-green-500',
        satisfactory: 'text-lime-500',
        moderate: 'text-yellow-500',
        poor: 'text-orange-500',
        'very-poor': 'text-red-500',
        severe: 'text-purple-600',
    };
    return textClasses[colorClass];
}

export function getAQIBorderClass(colorClass: AQIColorClass): string {
    const borderClasses: Record<AQIColorClass, string> = {
        good: 'border-l-green-500',
        satisfactory: 'border-l-lime-500',
        moderate: 'border-l-yellow-500',
        poor: 'border-l-orange-500',
        'very-poor': 'border-l-red-500',
        severe: 'border-l-purple-600',
    };
    return borderClasses[colorClass];
}

export function shouldShowHealthWarning(aqi: number): boolean {
    return aqi > 100;
}

export function getAQIEmoji(colorClass: AQIColorClass): string {
    const emojis: Record<AQIColorClass, string> = {
        good: '😊',
        satisfactory: '🙂',
        moderate: '😐',
        poor: '😷',
        'very-poor': '🤢',
        severe: '☠️',
    };
    return emojis[colorClass];
}

// Scenario options
export const SCENARIO_OPTIONS: ScenarioOption[] = [
    { value: 'normal', label: 'Normal Conditions', modifier: 1.0 },
    { value: 'high_traffic', label: 'High Traffic (+30%)', modifier: 1.3 },
    { value: 'industrial', label: 'Industrial Activity (+50%)', modifier: 1.5 },
    { value: 'weather_event', label: 'Rain Event (-30%)', modifier: 0.7 },
    { value: 'diwali', label: 'Festival/Diwali (2x)', modifier: 2.0 },
];

export function getScenarioModifier(scenario: Scenario): number {
    return SCENARIO_OPTIONS.find((s) => s.value === scenario)?.modifier ?? 1.0;
}

// Format time for display
export function formatTime(date: Date): string {
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true,
    });
}

// Format countdown
export function formatCountdown(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Generate mock predictions
export function generateMockPredictions(baseAqi: number, scenario: Scenario): HourlyPrediction[] {
    const modifier = getScenarioModifier(scenario);
    const predictions: HourlyPrediction[] = [];
    const now = new Date();

    for (let i = 0; i < 24; i++) {
        const hour = new Date(now.getTime() + i * 60 * 60 * 1000);
        const hourStr = hour.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });

        // Simulate daily variation
        const hourOfDay = hour.getHours();
        let variation = 1.0;
        if (hourOfDay >= 7 && hourOfDay <= 10) variation = 1.15;
        else if (hourOfDay >= 17 && hourOfDay <= 20) variation = 1.2;
        else if (hourOfDay >= 0 && hourOfDay <= 5) variation = 0.85;

        const randomFactor = 0.9 + Math.random() * 0.2;
        const aqi = Math.round(baseAqi * modifier * variation * randomFactor);
        const uncertainty = Math.round(aqi * 0.1);
        const colorClass = getAQIClass(aqi);

        predictions.push({
            hour: hourStr,
            aqi,
            status: getAQIStatus(aqi),
            color: colorClass,
            aqi_lower: aqi - uncertainty,
            aqi_upper: aqi + uncertainty,
            uncertainty,
        });
    }

    return predictions;
}

// Generate historical data for analytics
export function generateHistoricalData(city: string, days: number): HistoricalDataPoint[] {
    const data: HistoricalDataPoint[] = [];
    const now = new Date();

    // City-specific base AQI
    const cityBases: Record<string, number> = {
        Delhi: 280,
        Mumbai: 120,
        Bangalore: 85,
        Chennai: 75,
        Kolkata: 150,
        Hyderabad: 90,
        Chikkamagaluru: 45,
        Madikeri: 40,
    };
    const baseAqi = cityBases[city] || 100;

    for (let i = days - 1; i >= 0; i--) {
        const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
        const dateStr = date.toISOString().split('T')[0];

        // Add some seasonal variation
        const dayOfYear = Math.floor((date.getTime() - new Date(date.getFullYear(), 0, 0).getTime()) / (1000 * 60 * 60 * 24));
        const seasonalFactor = 1 + 0.3 * Math.sin((dayOfYear / 365) * 2 * Math.PI - Math.PI / 2); // Higher in winter

        // Add weekly variation (weekends slightly lower)
        const dayOfWeek = date.getDay();
        const weekendFactor = dayOfWeek === 0 || dayOfWeek === 6 ? 0.9 : 1.0;

        // Random daily variation
        const randomFactor = 0.8 + Math.random() * 0.4;

        const aqi = Math.round(baseAqi * seasonalFactor * weekendFactor * randomFactor);
        const pm2_5 = Math.round((aqi / 50) * 30 + Math.random() * 15);
        const pm10 = Math.round(pm2_5 * 1.5 + Math.random() * 25);
        const no2 = Math.round(20 + (aqi / 100) * 35 + Math.random() * 15);
        const so2 = Math.round(10 + (aqi / 100) * 18 + Math.random() * 10);
        const co = Math.round((5 + (aqi / 100) * 8 + Math.random() * 4) * 10) / 10;
        const o3 = Math.round(30 + (aqi / 100) * 25 + Math.random() * 20);

        data.push({
            date: dateStr,
            aqi,
            pm2_5,
            pm10,
            no2,
            so2,
            co,
            o3,
        });
    }

    return data;
}

// Debounce utility
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function debounce<T extends (...args: any[]) => void>(func: T, wait: number): (...args: Parameters<T>) => void {
    let timeout: ReturnType<typeof setTimeout> | null = null;

    return (...args: Parameters<T>) => {
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), wait);
    };
}

// Format date for display
export function formatDate(date: Date | string): string {
    const d = typeof date === 'string' ? new Date(date) : date;
    return d.toLocaleDateString('en-IN', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
    });
}

// Calculate statistics from historical data
export function calculateStats(data: HistoricalDataPoint[]): {
    avgAqi: number;
    maxAqi: number;
    minAqi: number;
    trend: 'improving' | 'stable' | 'worsening';
} {
    if (data.length === 0) {
        return { avgAqi: 0, maxAqi: 0, minAqi: 0, trend: 'stable' };
    }

    const aqiValues = data.map((d) => d.aqi);
    const avgAqi = Math.round(aqiValues.reduce((a, b) => a + b, 0) / aqiValues.length);
    const maxAqi = Math.max(...aqiValues);
    const minAqi = Math.min(...aqiValues);

    // Calculate trend from last 7 days vs previous 7 days
    if (data.length >= 14) {
        const recent = data.slice(-7).reduce((a, b) => a + b.aqi, 0) / 7;
        const previous = data.slice(-14, -7).reduce((a, b) => a + b.aqi, 0) / 7;
        const diff = recent - previous;
        if (diff > 10) return { avgAqi, maxAqi, minAqi, trend: 'worsening' };
        if (diff < -10) return { avgAqi, maxAqi, minAqi, trend: 'improving' };
    }

    return { avgAqi, maxAqi, minAqi, trend: 'stable' };
}

// Export data to CSV
export function exportToCSV(data: HistoricalDataPoint[], filename: string): void {
    const headers = ['Date', 'AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'];
    const csvContent =
        headers.join(',') +
        '\n' +
        data.map((row) => [row.date, row.aqi, row.pm2_5, row.pm10, row.no2, row.so2, row.co, row.o3].join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
