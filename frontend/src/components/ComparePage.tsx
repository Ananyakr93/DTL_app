
import { useState, useEffect, useCallback } from 'react';
import { useStore, ComparisonItem } from '../store';
import { fetchCurrentAQI } from '../api';
import { searchLocations } from '../data/cities';
import { AQIData } from '../types';
import { X, Search, Plus, MapPin, Download, AlertTriangle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { getAQIColor, debounce } from '../utils';

export default function ComparePage() {
    const { comparisonList, addToComparison, removeFromComparison, allStations, settings } = useStore();
    const isDarkMode = settings.isDarkMode;

    const [comparisonData, setComparisonData] = useState<Record<string, AQIData>>({});
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Search state
    const [searchValue, setSearchValue] = useState('');
    const [suggestions, setSuggestions] = useState<Array<{ type: 'city' | 'station'; name: string; city?: string; state: string; id?: string; stationData?: any }>>([]);
    const [showSuggestions, setShowSuggestions] = useState(false);

    // For generating CSV
    const downloadCSV = () => {
        const headers = ['Location', 'AQI', 'Status', 'PM2.5', 'PM10', 'NO2', 'Dominant Pollutant'];
        const rows = comparisonList.map(item => {
            const data = comparisonData[item.id];
            if (!data) return [item.name, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'];
            const c = data.current;
            return [
                item.name,
                c.aqi_value,
                c.aqi_status,
                c.pm2_5 || 'N/A',
                c.pm10 || 'N/A',
                c.no2 || 'N/A',
                c.dominant_pollutant
            ];
        });

        const csvContent = "data:text/csv;charset=utf-8,"
            + headers.join(",") + "\n"
            + rows.map(e => e.join(",")).join("\n");

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `aqi_comparison_${new Date().toISOString().split('T')[0]}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    // Fetch data for comparison list
    useEffect(() => {
        const fetchData = async () => {
            if (comparisonList.length === 0) return;

            setIsLoading(true);
            setError(null);

            const newData: Record<string, AQIData> = {};

            try {
                // Fetch in parallel
                await Promise.all(comparisonList.map(async (item) => {
                    // Avoid re-fetching if we already have fresh data (optional optimization, skipping for simplicity/freshness)
                    try {
                        const station = item.type === 'station' ? item.station : undefined;
                        // Use city name for fetch, or station name if available
                        const searchName = item.type === 'station' ? (item.station?.city || item.name.split(',')[0]) : item.name;

                        const data = await fetchCurrentAQI(searchName, station);
                        newData[item.id] = data;
                    } catch (e) {
                        console.error(`Failed to fetch for ${item.name}`, e);
                    }
                }));

                setComparisonData(newData);
            } catch (err) {
                setError('Failed to load comparison data');
            } finally {
                setIsLoading(false);
            }
        };

        if (comparisonList.length > 0) {
            fetchData();
        } else {
            setComparisonData({});
        }
    }, [comparisonList]);

    // Search Logic
    const debouncedSearch = useCallback(
        debounce(async (query: string) => {
            if (query.length >= 2) {
                const cityResults = searchLocations(query);

                // Search in allStations
                const stationResults = allStations
                    .filter((s: any) => s.station?.name?.toLowerCase().includes(query.toLowerCase()))
                    .map((s: any) => ({
                        type: 'station' as const,
                        name: s.station.name,
                        city: s.station.name.split(',')[0],
                        state: 'India',
                        id: String(s.uid),
                        stationData: s
                    }))
                    .slice(0, 5);

                // Simple merge logic
                const citySuggestions = cityResults.map(c => ({
                    type: 'city' as const,
                    name: c.name,
                    city: c.name,
                    state: c.state || 'India',
                    id: c.name
                })).slice(0, 3);

                setSuggestions([...citySuggestions, ...stationResults].slice(0, 8));
                setShowSuggestions(true);
            } else {
                setSuggestions([]);
                setShowSuggestions(false);
            }
        }, 300),
        [allStations]
    );

    const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
        setSearchValue(e.target.value);
        debouncedSearch(e.target.value);
    };

    const handleAdd = (suggestion: any) => {
        const newItem: ComparisonItem = {
            id: suggestion.id,
            name: suggestion.name,
            type: 'station', // Simplify to station for bounds results
            station: {
                id: String(suggestion.stationData.uid),
                name: suggestion.name,
                city: suggestion.city,
                state: suggestion.state,
                lat: suggestion.stationData.lat,
                lon: suggestion.stationData.lon
            }
        };
        addToComparison(newItem);
        setSearchValue('');
        setShowSuggestions(false);
    };

    // Chart Data Preparation
    const chartData = comparisonList.map(item => {
        const data = comparisonData[item.id];
        return {
            name: item.name.split(',')[0], // Short name
            aqi: data?.current?.aqi_value || 0,
            color: data?.current?.aqi_color || 'gray'
        };
    });

    return (
        <div className="space-y-6 animate-fade-in pb-10">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                        Compare Locations
                    </h1>
                    <p className={`mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        Compare air quality across {comparisonList.length}/5 locations
                    </p>
                </div>

                <div className="flex gap-3">
                    {comparisonList.length > 0 && (
                        <button
                            onClick={downloadCSV}
                            className={`flex items-center gap-2 px-4 py-2 rounded-xl border ${isDarkMode ? 'border-slate-700 text-gray-300 hover:bg-slate-800' : 'border-gray-200 text-gray-700 hover:bg-gray-50'
                                }`}
                        >
                            <Download className="w-4 h-4" />
                            Export CSV
                        </button>
                    )}
                </div>
            </div>

            {error && (
                <div className={`p-4 rounded-xl ${isDarkMode ? 'bg-red-900/30 text-red-400' : 'bg-red-100 text-red-700'} flex items-center gap-2`}>
                    <AlertTriangle className="w-5 h-5" />
                    {error}
                </div>
            )}

            {/* Search Bar */}
            <div className="relative max-w-lg">
                <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border-2 transition-all ${isDarkMode ? 'bg-slate-800 border-slate-700 focus-within:border-brand-primary' : 'bg-white border-gray-200 focus-within:border-brand-primary'
                    }`}>
                    <Search className={`w-5 h-5 ${isDarkMode ? 'text-gray-400' : 'text-gray-400'}`} />
                    <input
                        type="text"
                        placeholder="Search to add location..."
                        value={searchValue}
                        onChange={handleSearch}
                        disabled={comparisonList.length >= 5}
                        className={`flex-1 bg-transparent outline-none ${isDarkMode ? 'text-white placeholder-gray-500' : 'text-gray-900 placeholder-gray-400'}`}
                    />
                </div>

                {showSuggestions && suggestions.length > 0 && (
                    <div className={`absolute top-full left-0 right-0 mt-2 rounded-xl shadow-xl border overflow-hidden z-50 ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-gray-200'}`}>
                        {suggestions.map((s, idx) => (
                            <button
                                key={idx}
                                onClick={() => handleAdd(s)}
                                className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors ${isDarkMode ? 'hover:bg-slate-700 text-white' : 'hover:bg-gray-50 text-gray-900'
                                    }`}
                            >
                                <Plus className="w-4 h-4 text-brand-primary" />
                                <div>
                                    <p className="font-medium">{s.name}</p>
                                    <p className="text-xs opacity-60">{s.city}, {s.state}</p>
                                </div>
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {/* Comparison Cards */}
            {comparisonList.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
                    {comparisonList.map(item => {
                        const data = comparisonData[item.id];
                        const isLoadingData = !data && isLoading;

                        return (
                            <div key={item.id} className={`relative p-5 rounded-2xl border ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-gray-100'} shadow-sm`}>
                                <button
                                    onClick={() => removeFromComparison(item.id)}
                                    className="absolute top-2 right-2 p-1 rounded-full hover:bg-gray-200 dark:hover:bg-slate-600 opacity-50 hover:opacity-100 transition-all"
                                >
                                    <X className="w-4 h-4" />
                                </button>

                                <h3 className={`font-bold truncate pr-6 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{item.name}</h3>

                                <div className="mt-4">
                                    {isLoadingData ? (
                                        <div className="animate-pulse space-y-2">
                                            <div className="h-8 w-16 bg-gray-200 dark:bg-slate-700 rounded"></div>
                                            <div className="h-4 w-24 bg-gray-200 dark:bg-slate-700 rounded"></div>
                                        </div>
                                    ) : data ? (
                                        <>
                                            <div className="flex items-end gap-2">
                                                <span className="text-4xl font-bold" style={{ color: getAQIColor(data.current.aqi_color) }}>
                                                    {data.current.aqi_value}
                                                </span>
                                                <span className="text-sm font-medium mb-1" style={{ color: getAQIColor(data.current.aqi_color) }}>
                                                    {data.current.aqi_status}
                                                </span>
                                            </div>
                                            <p className={`text-xs mt-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                                Primary: {data.current.dominant_pollutant.toUpperCase()}
                                            </p>
                                        </>
                                    ) : (
                                        <div className="text-red-500 text-sm flex items-center gap-1">
                                            <AlertTriangle className="w-4 h-4" /> Failed to load
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            ) : (
                <div className={`text-center py-20 rounded-2xl border-2 border-dashed ${isDarkMode ? 'border-slate-800 text-gray-500' : 'border-gray-200 text-gray-400'}`}>
                    <MapPin className="w-12 h-12 mx-auto mb-3 opacity-20" />
                    <p className="text-lg font-medium">No locations selected</p>
                    <p className="text-sm">Search and add up to 5 locations to compare</p>
                </div>
            )}

            {/* Charts Section */}
            {comparisonList.length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Bar Chart */}
                    <div className={`lg:col-span-2 p-6 rounded-2xl shadow-lg ${isDarkMode ? 'bg-slate-800' : 'bg-white'}`}>
                        <h3 className={`text-lg font-bold mb-6 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>AQI Comparison</h3>
                        <div className="h-[300px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                    <XAxis dataKey="name" stroke={isDarkMode ? '#94a3b8' : '#64748b'} />
                                    <YAxis stroke={isDarkMode ? '#94a3b8' : '#64748b'} />
                                    <Tooltip
                                        contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                        cursor={{ fill: isDarkMode ? '#334155' : '#f1f5f9', opacity: 0.4 }}
                                    />
                                    <Bar dataKey="aqi" radius={[4, 4, 0, 0]}>
                                        {chartData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={getAQIColor(entry.color as any)} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Pollutant Table */}
                    <div className={`p-6 rounded-2xl shadow-lg ${isDarkMode ? 'bg-slate-800' : 'bg-white'}`}>
                        <h3 className={`text-lg font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Pollutant Details</h3>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className={`text-left ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                        <th className="pb-3 text-sm font-normal">Location</th>
                                        <th className="pb-3 text-sm font-normal text-right">PM2.5</th>
                                        <th className="pb-3 text-sm font-normal text-right">PM10</th>
                                    </tr>
                                </thead>
                                <tbody className={`divide-y ${isDarkMode ? 'divide-slate-700' : 'divide-gray-100'}`}>
                                    {comparisonList.map(item => {
                                        const data = comparisonData[item.id];
                                        if (!data) return null;
                                        return (
                                            <tr key={item.id}>
                                                <td className={`py-3 font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                                    {item.name.split(',')[0]}
                                                </td>
                                                <td className="py-3 text-right font-mono opacity-80">
                                                    {data.current.pm2_5 || '-'}
                                                </td>
                                                <td className="py-3 text-right font-mono opacity-80">
                                                    {data.current.pm10 || '-'}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

