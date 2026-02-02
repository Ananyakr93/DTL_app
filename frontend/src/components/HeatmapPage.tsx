import { useEffect, useState, useCallback } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { Map, Globe2, RefreshCw, Navigation, Locate, Info } from 'lucide-react';
import { useStore } from '../store';
import { fetchMultipleCityAQI } from '../api';
import { INDIAN_CITIES } from '../data/cities';
import { getAQIColor } from '../utils';
import type { CityAQI } from '../types';
import 'leaflet/dist/leaflet.css';

const INDIA_CENTER: [number, number] = [22.5, 82.5];
const INDIA_ZOOM = 5;

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

export default function HeatmapPage() {
    const { city: currentCity, cityAQIData, setCityAQIData, settings, setCity, setActivePage } = useStore();
    const isDarkMode = settings.isDarkMode;

    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [lastFetch, setLastFetch] = useState<Date | null>(null);
    const [selectedCityData, setSelectedCityData] = useState<CityAQI | null>(null);

    const loadCityAQI = useCallback(async () => {
        if (isLoading) return;

        setIsLoading(true);
        setError(null);

        try {
            console.log('Loading AQI data for', INDIAN_CITIES.length, 'cities...');
            const data = await fetchMultipleCityAQI(INDIAN_CITIES);
            setCityAQIData(data);
            setLastFetch(new Date());
            console.log('Loaded AQI data for', data.length, 'cities');
        } catch (err) {
            console.error('Error fetching city AQI data:', err);
            setError('Failed to load map data. Please try again.');
        } finally {
            setIsLoading(false);
        }
    }, [isLoading, setCityAQIData]);

    useEffect(() => {
        if (cityAQIData.length === 0) {
            loadCityAQI();
        }
    }, [cityAQIData.length, loadCityAQI]);

    const getMarkerRadius = (aqi: number): number => {
        if (aqi <= 50) return 10;
        if (aqi <= 100) return 12;
        if (aqi <= 200) return 14;
        if (aqi <= 300) return 16;
        return 18;
    };

    const handleCityClick = (cityData: CityAQI) => {
        setSelectedCityData(cityData);
    };

    const handleViewDashboard = (cityName: string) => {
        setCity(cityName);
        setActivePage('dashboard');
    };

    // Stats
    const avgAqi = cityAQIData.length > 0
        ? Math.round(cityAQIData.reduce((sum, c) => sum + c.aqi, 0) / cityAQIData.length)
        : 0;
    const cleanestCity = cityAQIData.length > 0
        ? cityAQIData.reduce((min, c) => c.aqi < min.aqi ? c : min, cityAQIData[0])
        : null;
    const mostPollutedCity = cityAQIData.length > 0
        ? cityAQIData.reduce((max, c) => c.aqi > max.aqi ? c : max, cityAQIData[0])
        : null;

    return (
        <div className="space-y-6 animate-fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className={`text-2xl font-bold flex items-center gap-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                        <Globe2 className="w-8 h-8 text-brand-primary" />
                        India AQI Heatmap
                    </h1>
                    <p className={`mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        Real-time air quality across {cityAQIData.length} cities • Click any city to view details
                    </p>
                </div>

                <div className="flex items-center gap-3">
                    {lastFetch && (
                        <span className={`text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                            Updated: {lastFetch.toLocaleTimeString()}
                        </span>
                    )}
                    <button
                        onClick={loadCityAQI}
                        disabled={isLoading}
                        className="flex items-center gap-2 px-4 py-2 bg-brand-primary text-brand-dark rounded-xl font-medium hover:bg-opacity-90 transition-colors disabled:opacity-50"
                    >
                        <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                        {isLoading ? 'Loading...' : 'Refresh Data'}
                    </button>
                </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className={`p-5 rounded-2xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                    <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center">
                            <Map className="w-6 h-6 text-blue-500" />
                        </div>
                        <div>
                            <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Average AQI</p>
                            <p className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{avgAqi}</p>
                        </div>
                    </div>
                </div>

                {cleanestCity && (
                    <div className={`p-5 rounded-2xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                        <div className="flex items-center gap-3">
                            <div className="w-12 h-12 rounded-xl bg-green-500/20 flex items-center justify-center">
                                <Navigation className="w-6 h-6 text-green-500" />
                            </div>
                            <div>
                                <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Cleanest City</p>
                                <p className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                    {cleanestCity.name}
                                </p>
                                <p className="text-sm text-green-500 font-medium">AQI: {cleanestCity.aqi}</p>
                            </div>
                        </div>
                    </div>
                )}

                {mostPollutedCity && (
                    <div className={`p-5 rounded-2xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                        <div className="flex items-center gap-3">
                            <div className="w-12 h-12 rounded-xl bg-red-500/20 flex items-center justify-center">
                                <Locate className="w-6 h-6 text-red-500" />
                            </div>
                            <div>
                                <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Most Polluted</p>
                                <p className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                    {mostPollutedCity.name}
                                </p>
                                <p className="text-sm text-red-500 font-medium">AQI: {mostPollutedCity.aqi}</p>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Main Map Container */}
            <div className={`rounded-2xl overflow-hidden shadow-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'}`}>
                {/* Legend */}
                <div className={`p-4 border-b ${isDarkMode ? 'border-slate-700' : 'border-gray-200'}`}>
                    <div className="flex flex-wrap gap-4">
                        {[
                            { label: 'Good (0-50)', color: '#22c55e' },
                            { label: 'Satisfactory (51-100)', color: '#84cc16' },
                            { label: 'Moderate (101-200)', color: '#eab308' },
                            { label: 'Poor (201-300)', color: '#f97316' },
                            { label: 'Very Poor (301-400)', color: '#ef4444' },
                            { label: 'Severe (400+)', color: '#9333ea' },
                        ].map((item) => (
                            <div key={item.label} className="flex items-center gap-2 text-sm">
                                <span className="w-4 h-4 rounded-full" style={{ backgroundColor: item.color }} />
                                <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>{item.label}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Error */}
                {error && (
                    <div className="p-4 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400">
                        {error}
                    </div>
                )}

                {/* Map */}
                <div className="h-[600px] relative">
                    {isLoading && cityAQIData.length === 0 && (
                        <div className="absolute inset-0 flex items-center justify-center bg-slate-100 dark:bg-slate-700 z-10">
                            <div className="flex flex-col items-center gap-3">
                                <RefreshCw className="w-10 h-10 animate-spin text-brand-primary" />
                                <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                                    Loading AQI data for {INDIAN_CITIES.length} cities...
                                </span>
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

                        {cityAQIData.map((cityData) => (
                            <CircleMarker
                                key={cityData.name}
                                center={[cityData.lat, cityData.lon]}
                                radius={getMarkerRadius(cityData.aqi)}
                                pathOptions={{
                                    fillColor: getAQIColor(cityData.color),
                                    fillOpacity: 0.85,
                                    color: currentCity === cityData.name ? '#fff' : getAQIColor(cityData.color),
                                    weight: currentCity === cityData.name ? 3 : 1,
                                }}
                                eventHandlers={{
                                    click: () => handleCityClick(cityData),
                                }}
                            >
                                <Popup>
                                    <div className="text-center min-w-[150px] p-2">
                                        <p className="font-bold text-lg text-gray-900">{cityData.name}</p>
                                        <p className="text-sm text-gray-500 mb-2">{cityData.state}</p>
                                        <div className="my-3">
                                            <span
                                                className="text-4xl font-bold"
                                                style={{ color: getAQIColor(cityData.color) }}
                                            >
                                                {cityData.aqi}
                                            </span>
                                            <p className="text-sm font-medium mt-1" style={{ color: getAQIColor(cityData.color) }}>
                                                {cityData.status}
                                            </p>
                                        </div>
                                        <button
                                            onClick={() => handleViewDashboard(cityData.name)}
                                            className="w-full mt-2 px-4 py-2 bg-brand-primary text-brand-dark text-sm font-medium rounded-lg hover:bg-opacity-80 transition-colors"
                                        >
                                            View Dashboard →
                                        </button>
                                    </div>
                                </Popup>
                            </CircleMarker>
                        ))}
                    </MapContainer>
                </div>

                {/* Footer */}
                <div className={`p-4 ${isDarkMode ? 'bg-slate-700/50' : 'bg-gray-50'} flex items-center justify-between`}>
                    <div className="flex items-center gap-2 text-sm">
                        <Info className="w-4 h-4" />
                        <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
                            Data sources: CPCB, WAQI • Click markers to view city details
                        </span>
                    </div>
                    <span className={`text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                        {cityAQIData.length} cities monitored
                    </span>
                </div>
            </div>

            {/* Selected City Details */}
            {selectedCityData && (
                <div className={`p-6 rounded-2xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-xl`}>
                    <div className="flex items-start justify-between">
                        <div>
                            <h3 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                {selectedCityData.name}
                            </h3>
                            <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                {selectedCityData.state}
                            </p>
                        </div>
                        <div className="text-right">
                            <p
                                className="text-4xl font-bold"
                                style={{ color: getAQIColor(selectedCityData.color) }}
                            >
                                {selectedCityData.aqi}
                            </p>
                            <p
                                className="text-sm font-medium"
                                style={{ color: getAQIColor(selectedCityData.color) }}
                            >
                                {selectedCityData.status}
                            </p>
                        </div>
                    </div>
                    <div className="mt-6 flex gap-4">
                        <button
                            onClick={() => handleViewDashboard(selectedCityData.name)}
                            className="flex-1 px-4 py-3 bg-brand-primary text-brand-dark font-medium rounded-xl hover:bg-opacity-90 transition-colors"
                        >
                            View Full Dashboard
                        </button>
                        <button
                            onClick={() => setSelectedCityData(null)}
                            className={`px-4 py-3 rounded-xl font-medium ${isDarkMode ? 'bg-slate-700 text-gray-300' : 'bg-gray-100 text-gray-700'
                                }`}
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
