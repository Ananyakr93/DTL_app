// AQI Color Classes matching CPCB India standard
export type AQIColorClass = 'good' | 'satisfactory' | 'moderate' | 'poor' | 'very-poor' | 'severe';

// Scenario types for predictions
export type Scenario = 'normal' | 'high_traffic' | 'industrial' | 'weather_event' | 'diwali';

export interface ScenarioOption {
    value: Scenario;
    label: string;
    modifier: number;
}

// Health condition types for personalized recommendations
export type HealthCondition =
    | 'asthma'
    | 'respiratory'
    | 'heart_disease'
    | 'pregnant'
    | 'elderly'
    | 'children'
    | 'none';

// User settings including health profile
export interface UserSettings {
    isDarkMode: boolean;
    defaultCity: string;
    units: 'metric' | 'imperial';
    alertThreshold: number;
    enableNotifications: boolean;
    refreshInterval: number;
    healthConditions: HealthCondition[];
}

// Current AQI data from API
export interface CurrentAQI {
    aqi_value: number;
    aqi_status: string;
    aqi_color: AQIColorClass;
    station: string;
    aqi_source: string;
    dominant_pollutant: string;
    pm2_5: number | null;
    pm10: number | null;
    no2: number | null;
    so2: number | null;
    co: number | null;
    o3: number | null;
    temperature?: number | null;
    humidity?: number | null;
    wind?: number | null;
    timestamp?: string;
}

// Health recommendations
export interface HealthRecommendations {
    general: string;
    sensitive: string;
    outdoor_advice: string;
    mask_advice: string;
    specific_advice: Array<{
        pollutant: string;
        message: string;
        action: string;
    }>;
}

// Personal risk assessment
export interface PersonalRisk {
    risk_level: 'Low' | 'Moderate' | 'High' | 'Severe';
    alert_title: string;
    alert_message: string;
    tips: string[];
    personalized_warnings?: string[];
}

// Complete AQI data response
export interface AQIData {
    city: string;
    current: CurrentAQI;
    health: HealthRecommendations;
    activities: string[];
    personal_risk: PersonalRisk;
}

// Hourly prediction data
export interface HourlyPrediction {
    hour: string;
    aqi: number;
    status: string;
    color: AQIColorClass;
    aqi_lower?: number;
    aqi_upper?: number;
    uncertainty?: number;
}

// Historical data point
export interface HistoricalDataPoint {
    date: string;
    aqi: number;
    pm2_5: number;
    pm10: number;
    no2: number;
    so2: number;
    co: number;
    o3: number;
}

// Page types for routing
export type PageType = 'dashboard' | 'heatmap' | 'analytics' | 'reports' | 'settings';

// City AQI data for heatmap
export interface CityAQI {
    name: string;
    state: string;
    lat: number;
    lon: number;
    aqi: number;
    status: string;
    color: AQIColorClass;
}

// WAQI API Response types
export interface WAQIResponse {
    status: string;
    data: {
        aqi: number;
        idx: number;
        attributions: Array<{
            url: string;
            name: string;
        }>;
        city: {
            geo: [number, number];
            name: string;
            url: string;
        };
        dominentpol?: string;
        iaqi: {
            pm25?: { v: number };
            pm10?: { v: number };
            no2?: { v: number };
            so2?: { v: number };
            co?: { v: number };
            o3?: { v: number };
            t?: { v: number };
            h?: { v: number };
            w?: { v: number };
        };
        time: {
            s: string;
            tz: string;
            v: number;
        };
        forecast?: {
            daily: {
                pm25?: Array<{ avg: number; day: string; max: number; min: number }>;
                pm10?: Array<{ avg: number; day: string; max: number; min: number }>;
                o3?: Array<{ avg: number; day: string; max: number; min: number }>;
            };
        };
    };
}
