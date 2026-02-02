import { useEffect, useState, useCallback } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { Map, Globe2, RefreshCw, Navigation, Locate, Info } from 'lucide-react';
import { useStore } from '../store';
import { fetchStationsInBounds } from '../api';
import { INDIAN_CITIES } from '../data/cities';
import { getAQIColor, getAQIStatus, getAQIClass } from '../utils';

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
    const { city: currentCity, allStations, setAllStations, settings, setCity, setActivePage } = useStore();
    const isDarkMode = settings.isDarkMode;

    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [lastFetch, setLastFetch] = useState<Date | null>(null);
    const [selectedStation, setSelectedStation] = useState<any | null>(null);

    const loadMapData = useCallback(async () => {
        if (isLoading) return;

        setIsLoading(true);
        setError(null);

        try {
            console.log('Loading full station data for India bounds...');
            const data = await fetchStationsInBounds();
            if (data.length > 0) {
                setAllStations(data);
                setLastFetch(new Date());
                console.log('Loaded', data.length, 'stations');
            } else {
                setError('No station data found in bounds');
            }
        } catch (err) {
            console.error('Error fetching station data:', err);
            setError('Failed to load map data. Please try again.');
        } finally {
            setIsLoading(false);
        }
    }, [isLoading, setAllStations]);

    useEffect(() => {
        if (allStations.length === 0) {
            loadMapData();
        }
    }, [allStations.length, loadMapData]);

    const getMarkerRadius = (aqi: number): number => {
        if (aqi <= 50) return 10;
        if (aqi <= 100) return 12;
        if (aqi <= 200) return 14;
        if (aqi <= 300) return 16;
        return 18;
    };

    const handleStationClick = (station: any) => {
        setSelectedStation(station);
    };

    const handleViewDashboard = (cityName: string) => {
        setCity(cityName); // Note: Should we set station too? For now, city is fine.
        setActivePage('dashboard');
    };

    // Stats
    const validStations = allStations.filter(s => !isNaN(Number(s.aqi)));
    const avgAqi = validStations.length > 0
        ? Math.round(validStations.reduce((sum, s) => sum + Number(s.aqi), 0) / validStations.length)
        : 0;
    const cleanest = validStations.length > 0
        ? validStations.reduce((min, s) => Number(s.aqi) < Number(min.aqi) ? s : min, validStations[0])
        : null;
    const mostPolluted = validStations.length > 0
        ? validStations.reduce((max, s) => Number(s.aqi) > Number(max.aqi) ? s : max, validStations[0])
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
                        Real-time air quality from {allStations.length} stations across India
                    </p>
                </div>

                <div className="flex items-center gap-3">
                    {lastFetch && (
                        <span className={`text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                            Updated: {lastFetch.toLocaleTimeString()}
                        </span>
                    )}
                    <button
                        onClick={loadMapData}
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

                {cleanest && (
                    <div className={`p-5 rounded-2xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                        <div className="flex items-center gap-3">
                            <div className="w-12 h-12 rounded-xl bg-green-500/20 flex items-center justify-center">
                                <Navigation className="w-6 h-6 text-green-500" />
                            </div>
                            <div>
                                <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Cleanest Location</p>
                                <p className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'} line-clamp-1`} title={cleanest?.station.name}>
                                    {cleanest?.station.name}
                                </p>
                                <p className="text-sm text-green-500 font-medium">AQI: {cleanest?.aqi}</p>
                            </div>
                        </div>
                    </div>
                )}

                {mostPolluted && (
                    <div className={`p-5 rounded-2xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                        <div className="flex items-center gap-3">
                            <div className="w-12 h-12 rounded-xl bg-red-500/20 flex items-center justify-center">
                                <Locate className="w-6 h-6 text-red-500" />
                            </div>
                            <div>
                                <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Most Polluted</p>
                                <p className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'} line-clamp-1`} title={mostPolluted?.station.name}>
                                    {mostPolluted?.station.name}
                                </p>
                                <p className="text-sm text-red-500 font-medium">AQI: {mostPolluted?.aqi}</p>
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
                    {isLoading && allStations.length === 0 && (
                        <div className="absolute inset-0 flex items-center justify-center bg-slate-100 dark:bg-slate-700 z-10">
                            <div className="flex flex-col items-center gap-3">
                                <RefreshCw className="w-10 h-10 animate-spin text-brand-primary" />
                                <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                                    Loading full station map for India...
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

                        {allStations.map((station) => {
                            const aqi = Number(station.aqi);
                            if (isNaN(aqi)) return null;
                            const colorClass = getAQIClass(aqi);
                            const status = getAQIStatus(aqi);
                            const color = getAQIColor(colorClass);

                            return (
                                <CircleMarker
                                    key={station.uid}
                                    center={[station.lat, station.lon]}
                                    radius={getMarkerRadius(aqi)}
                                    pathOptions={{
                                        fillColor: color,
                                        fillOpacity: 0.8,
                                        color: selectedStation?.uid === station.uid ? '#fff' : color,
                                        weight: selectedStation?.uid === station.uid ? 3 : 1,
                                    }}
                                    eventHandlers={{
                                        click: () => handleStationClick(station),
                                    }}
                                >
                                    <Popup>
                                        <div className="text-center min-w-[180px] p-2">
                                            <p className="font-bold text-base text-gray-900 break-words">{station.station.name}</p>
                                            <p className="text-xs text-gray-500 mb-2">Updated: {station.station.time}</p>
                                            <div className="my-3">
                                                <span
                                                    className="text-4xl font-bold"
                                                    style={{ color: color }}
                                                >
                                                    {station.aqi}
                                                </span>
                                                <p className="text-sm font-medium mt-1" style={{ color: color }}>
                                                    {status}
                                                </p>
                                            </div>
                                            <button
                                                onClick={() => handleViewDashboard(station.station.name.split(',')[0])}
                                                className="w-full mt-2 px-3 py-2 bg-brand-primary text-brand-dark text-xs font-bold uppercase rounded-lg hover:bg-opacity-80 transition-colors"
                                            >
                                                View Detailed Analysis
                                            </button>
                                        </div>
                                    </Popup>
                                </CircleMarker>
                            );
                        })}
                    </MapContainer>
                </div>

                {/* Footer */}
                <div className={`p-4 ${isDarkMode ? 'bg-slate-700/50' : 'bg-gray-50'} flex items-center justify-between`}>
                    <div className="flex items-center gap-2 text-sm">
                        <Info className="w-4 h-4" />
                        <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
                            Data source: WAQI (Real-time)
                        </span>
                    </div>
                    <span className={`text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                        {validStations.length} stations monitored
                    </span>
                </div>
            </div>

            {/* Selected Station Details Panel */}
            {selectedStation && (
                <div className={`p-6 rounded-2xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-xl transition-all`}>
                    <div className="flex items-start justify-between">
                        <div className="flex-1">
                            <h3 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'} break-words`}>
                                {selectedStation.station.name}
                            </h3>
                            <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'} mt-1`}>
                                {selectedStation.lat.toFixed(4)}, {selectedStation.lon.toFixed(4)}
                            </p>
                        </div>
                        <div className="text-right ml-4">
                            <p
                                className="text-4xl font-bold"
                                style={{ color: getAQIColor(getAQIClass(Number(selectedStation.aqi))) }}
                            >
                                {selectedStation.aqi}
                            </p>
                            <p
                                className="text-sm font-medium"
                                style={{ color: getAQIColor(getAQIClass(Number(selectedStation.aqi))) }}
                            >
                                {getAQIStatus(Number(selectedStation.aqi))}
                            </p>
                        </div>
                    </div>
                    <div className="mt-6 flex gap-4">
                        <button
                            onClick={() => handleViewDashboard(selectedStation.station.name.split(',')[0])}
                            className="flex-1 px-4 py-3 bg-brand-primary text-brand-dark font-medium rounded-xl hover:bg-opacity-90 transition-colors"
                        >
                            View Full Dashboard
                        </button>
                        <button
                            onClick={() => setSelectedStation(null)}
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
