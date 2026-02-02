import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
    AQIData,
    HourlyPrediction,
    Scenario,
    PageType,
    UserSettings,
    HistoricalDataPoint,
    CityAQI,
    HealthCondition,
} from './types';
import type { Station } from './data/cities';

export interface ComparisonItem {
    id: string;
    name: string;
    type: 'city' | 'station';
    station?: Station;
}

interface AppState {
    // Location state
    city: string;
    setCity: (city: string) => void;
    selectedStation: Station | null;
    setSelectedStation: (station: Station | null) => void;

    // AQI data state
    currentData: AQIData | null;
    setCurrentData: (data: AQIData | null) => void;
    predictions: HourlyPrediction[];
    setPredictions: (predictions: HourlyPrediction[]) => void;
    historicalData: HistoricalDataPoint[];
    setHistoricalData: (data: HistoricalDataPoint[]) => void;

    // City AQI data for heatmap
    cityAQIData: CityAQI[];
    setCityAQIData: (data: CityAQI[]) => void;

    // Full station list (WAQI Bounds)
    allStations: any[]; // Using lax type for flexibility, ideally define strict type
    setAllStations: (stations: any[]) => void;

    // Scenario for predictions
    scenario: Scenario;
    setScenario: (scenario: Scenario) => void;

    // UI state
    isLoading: boolean;
    setIsLoading: (loading: boolean) => void;
    error: string | null;
    setError: (error: string | null) => void;
    activePage: PageType;
    setActivePage: (page: PageType) => void;

    // Comparison state
    comparisonList: ComparisonItem[];
    addToComparison: (item: ComparisonItem) => void;
    removeFromComparison: (id: string) => void;
    clearComparison: () => void;

    // Update timing
    lastUpdate: Date | null;
    setLastUpdate: (date: Date) => void;
    refreshCountdown: number;
    setRefreshCountdown: (seconds: number) => void;

    // User settings (persisted)
    settings: UserSettings;
    updateSettings: (settings: Partial<UserSettings>) => void;

    // Health profile helpers
    addHealthCondition: (condition: HealthCondition) => void;
    removeHealthCondition: (condition: HealthCondition) => void;
    hasHealthCondition: (condition: HealthCondition) => boolean;
}

const DEFAULT_SETTINGS: UserSettings = {
    isDarkMode: false,
    defaultCity: 'Bangalore',
    units: 'metric',
    alertThreshold: 100,
    enableNotifications: true,
    refreshInterval: 60,
    healthConditions: ['none'],
};

export const useStore = create<AppState>()(
    persist(
        (set, get) => ({
            // Location state
            city: 'Bangalore',
            setCity: (city) => set({ city }),
            selectedStation: null,
            setSelectedStation: (station) => set({ selectedStation: station }),

            // AQI data state
            currentData: null,
            setCurrentData: (data) => set({ currentData: data }),
            predictions: [],
            setPredictions: (predictions) => set({ predictions }),
            historicalData: [],
            setHistoricalData: (data) => set({ historicalData: data }),

            // City AQI data for heatmap
            cityAQIData: [],
            setCityAQIData: (data) => set({ cityAQIData: data }),

            allStations: [],
            setAllStations: (stations) => set({ allStations: stations }),

            // Scenario
            scenario: 'normal',
            setScenario: (scenario) => set({ scenario }),

            // UI state
            isLoading: false,
            setIsLoading: (loading) => set({ isLoading: loading }),
            error: null,
            setError: (error) => set({ error }),
            activePage: 'dashboard',
            setActivePage: (page) => set({ activePage: page }),

            // Comparison
            comparisonList: [],
            addToComparison: (item) => set((state) => {
                if (state.comparisonList.some(i => i.id === item.id)) return state;
                if (state.comparisonList.length >= 5) return state;
                return { comparisonList: [...state.comparisonList, item] };
            }),
            removeFromComparison: (id) => set((state) => ({
                comparisonList: state.comparisonList.filter(i => i.id !== id)
            })),
            clearComparison: () => set({ comparisonList: [] }),

            // Update timing
            lastUpdate: null,
            setLastUpdate: (date) => set({ lastUpdate: date }),
            refreshCountdown: 60,
            setRefreshCountdown: (seconds) => set({ refreshCountdown: seconds }),

            // User settings
            settings: DEFAULT_SETTINGS,
            updateSettings: (newSettings) =>
                set((state) => ({
                    settings: { ...state.settings, ...newSettings },
                })),

            // Health profile helpers
            addHealthCondition: (condition) =>
                set((state) => {
                    let conditions: HealthCondition[] = state.settings.healthConditions.filter((c) => c !== 'none');
                    if (condition === 'none') {
                        conditions = ['none'] as HealthCondition[];
                    } else if (!conditions.includes(condition)) {
                        conditions.push(condition);
                    }
                    return {
                        settings: { ...state.settings, healthConditions: conditions },
                    };
                }),

            removeHealthCondition: (condition) =>
                set((state) => {
                    const conditions = state.settings.healthConditions.filter((c) => c !== condition);
                    return {
                        settings: {
                            ...state.settings,
                            healthConditions: conditions.length === 0 ? ['none'] : conditions,
                        },
                    };
                }),

            hasHealthCondition: (condition) => {
                return get().settings.healthConditions.includes(condition);
            },
        }),
        {
            name: 'aeroclean-storage',
            partialize: (state) => ({
                settings: state.settings,
                city: state.city,
                comparisonList: state.comparisonList, // Persist comparison list
            }),
        }
    )
);
