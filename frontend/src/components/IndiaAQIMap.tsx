import { useEffect, useState, useCallback } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { Map, Globe2, RefreshCw } from 'lucide-react';
import { useStore } from '../store';
import { fetchMultipleCityAQI } from '../api';
import { INDIAN_CITIES } from '../data/cities';
import { getAQIColor } from '../utils';
import type { CityAQI } from '../types';
import 'leaflet/dist/leaflet.css';

// Map center and bounds for India
const INDIA_CENTER: [number, number] = [22.5, 82.5];
const INDIA_ZOOM = 5;

// Component to handle map events and updates
function MapController({ selectedCity }: { selectedCity: string | null }) {
    const map = useMap();

    useEffect(() => {
        if (selectedCity) {
            const city = INDIAN_CITIES.find((c) => c.name === selectedCity);
            if (city) {
                map.flyTo([city.lat, city.lon], 8, { duration: 1.5 });
            }
        }
    }, [selectedCity, map]);

    return null;
}

interface IndiaAQIMapProps {
    onCitySelect: (cityName: string) => void;
}

export default function IndiaAQIMap({ onCitySelect }: IndiaAQIMapProps) {
    const { city: currentCity, cityAQIData, setCityAQIData, settings } = useStore();
    const isDarkMode = settings.isDarkMode;

    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [lastFetch, setLastFetch] = useState<Date | null>(null);

    // Fetch AQI data for all cities
    const loadCityAQI = useCallback(async () => {
        if (isLoading) return;

        setIsLoading(true);
        setError(null);

        try {
            const data = await fetchMultipleCityAQI(INDIAN_CITIES);
            setCityAQIData(data);
            setLastFetch(new Date());
        } catch (err) {
            console.error('Error fetching city AQI data:', err);
            setError('Failed to load map data');
        } finally {
            setIsLoading(false);
        }
    }, [isLoading, setCityAQIData]);

    // Load data on mount
    useEffect(() => {
        if (cityAQIData.length === 0) {
            loadCityAQI();
        }
    }, [cityAQIData.length, loadCityAQI]);

    // Get marker size based on AQI
    const getMarkerRadius = (aqi: number): number => {
        if (aqi <= 50) return 8;
        if (aqi <= 100) return 10;
        if (aqi <= 200) return 12;
        if (aqi <= 300) return 14;
        return 16;
    };

    // Handle city click
    const handleCityClick = (cityData: CityAQI) => {
        onCitySelect(cityData.name);
    };

    return (
        <div className={`rounded-2xl overflow-hidden shadow-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'}`}>
            {/* Header */}
            <div className="p-4 border-b border-gray-200 dark:border-slate-700">
                <div className="flex items-center justify-between">
                    <h3 className={`text-lg font-bold flex items-center gap-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                        <Globe2 className="w-5 h-5 text-brand-primary" />
                        India AQI Heatmap
                    </h3>

                    <div className="flex items-center gap-3">
                        {lastFetch && (
                            <span className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                                Updated: {lastFetch.toLocaleTimeString()}
                            </span>
                        )}
                        <button
                            onClick={loadCityAQI}
                            disabled={isLoading}
                            className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${isDarkMode
                                    ? 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                } disabled:opacity-50`}
                        >
                            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                            {isLoading ? 'Loading...' : 'Refresh'}
                        </button>
                    </div>
                </div>

                {/* Legend */}
                <div className="flex flex-wrap gap-3 mt-3">
                    {[
                        { label: 'Good (0-50)', color: '#22c55e' },
                        { label: 'Satisfactory (51-100)', color: '#84cc16' },
                        { label: 'Moderate (101-200)', color: '#eab308' },
                        { label: 'Poor (201-300)', color: '#f97316' },
                        { label: 'Very Poor (301-400)', color: '#ef4444' },
                        { label: 'Severe (400+)', color: '#9333ea' },
                    ].map((item) => (
                        <div key={item.label} className="flex items-center gap-1.5 text-xs">
                            <span
                                className="w-3 h-3 rounded-full"
                                style={{ backgroundColor: item.color }}
                            />
                            <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>{item.label}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Error message */}
            {error && (
                <div className="p-3 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 text-sm">
                    {error}
                </div>
            )}

            {/* Map Container */}
            <div className="h-[400px] relative">
                {isLoading && cityAQIData.length === 0 && (
                    <div className="absolute inset-0 flex items-center justify-center bg-slate-100 dark:bg-slate-700 z-10">
                        <div className="flex flex-col items-center gap-3">
                            <RefreshCw className="w-8 h-8 animate-spin text-brand-primary" />
                            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Loading map data...</span>
                        </div>
                    </div>
                )}

                <MapContainer
                    center={INDIA_CENTER}
                    zoom={INDIA_ZOOM}
                    style={{ height: '100%', width: '100%' }}
                    scrollWheelZoom={true}
                    className="z-0"
                >
                    <TileLayer
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                        url={
                            isDarkMode
                                ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
                                : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
                        }
                    />

                    <MapController selectedCity={currentCity} />

                    {/* City markers */}
                    {cityAQIData.map((cityData) => (
                        <CircleMarker
                            key={cityData.name}
                            center={[cityData.lat, cityData.lon]}
                            radius={getMarkerRadius(cityData.aqi)}
                            pathOptions={{
                                fillColor: getAQIColor(cityData.color),
                                fillOpacity: 0.8,
                                color: currentCity === cityData.name ? '#fff' : getAQIColor(cityData.color),
                                weight: currentCity === cityData.name ? 3 : 1,
                            }}
                            eventHandlers={{
                                click: () => handleCityClick(cityData),
                            }}
                        >
                            <Popup>
                                <div className="text-center min-w-[120px]">
                                    <p className="font-bold text-gray-900">{cityData.name}</p>
                                    <p className="text-xs text-gray-500">{cityData.state}</p>
                                    <div className="mt-2">
                                        <span
                                            className="text-2xl font-bold"
                                            style={{ color: getAQIColor(cityData.color) }}
                                        >
                                            {cityData.aqi}
                                        </span>
                                        <p className="text-xs font-medium" style={{ color: getAQIColor(cityData.color) }}>
                                            {cityData.status}
                                        </p>
                                    </div>
                                    <button
                                        onClick={() => handleCityClick(cityData)}
                                        className="mt-2 px-3 py-1 bg-brand-primary text-brand-dark text-xs font-medium rounded-full hover:bg-opacity-80"
                                    >
                                        View Details
                                    </button>
                                </div>
                            </Popup>
                        </CircleMarker>
                    ))}
                </MapContainer>
            </div>

            {/* Click hint */}
            <div className={`p-3 text-center text-sm ${isDarkMode ? 'bg-slate-700/50 text-gray-400' : 'bg-gray-50 text-gray-500'}`}>
                <Map className="w-4 h-4 inline mr-1" />
                Click on any city marker to view detailed AQI information
            </div>
        </div>
    );
}
