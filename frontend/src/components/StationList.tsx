import { useState, useEffect } from 'react';
import { MapPin } from 'lucide-react';
import { useStore } from '../store';
import { searchCities } from '../api';
import { getAQIClass, getAQIStatus } from '../utils';

interface StationListProps {
    city: string;
}

interface StationSummary {
    name: string;
    aqi: number;
    status: string;
    color: string;
}

export default function StationList({ city }: StationListProps) {
    const { allStations } = useStore();
    const [stations, setStations] = useState<StationSummary[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let isMounted = true;

        async function loadStations() {
            if (!city) return;

            setIsLoading(true);
            setError(null);

            try {
                // Optimization: Use allStations from store if available
                if (allStations.length > 0) {
                    const normalizedCity = city.toLowerCase();
                    const filtered = allStations.filter(s =>
                        s.station.name.toLowerCase().includes(normalizedCity)
                    );

                    if (filtered.length > 0) {
                        const mapped = filtered
                            .filter(s => !isNaN(Number(s.aqi)))
                            .map(s => ({
                                name: s.station.name,
                                aqi: Number(s.aqi),
                                status: getAQIStatus(Number(s.aqi)),
                                color: getAQIClass(Number(s.aqi))
                            }));

                        if (isMounted) {
                            // Limit to 50 to match previous behavior/perf
                            setStations(mapped.slice(0, 50));
                            setIsLoading(false);
                            return;
                        }
                    }
                }

                // Fallback to API if not found in store or empty store
                const results = await searchCities(city);

                if (isMounted) {
                    if (results.length === 0) {
                        setStations([]);
                    } else {
                        // Filter and map results
                        const validStations = results
                            .filter(s => s.aqi > 0) // Only valid readings
                            .map(s => ({
                                name: s.name,
                                aqi: s.aqi,
                                status: getAQIStatus(s.aqi),
                                color: getAQIClass(s.aqi)
                            }));

                        setStations(validStations);
                    }
                }
            } catch (err) {
                if (isMounted) {
                    console.error("Failed to load stations:", err);
                    setError("Failed to load station list");
                }
            } finally {
                if (isMounted) {
                    setIsLoading(false);
                }
            }
        }

        loadStations();

        return () => { isMounted = false; };
    }, [city]);

    if (isLoading) {
        return (
            <div className="p-6 bg-white dark:bg-slate-800 rounded-xl shadow-lg animate-pulse">
                <div className="h-6 bg-gray-200 dark:bg-slate-700 rounded w-1/3 mb-4"></div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {[1, 2, 3, 4, 5, 6].map(i => (
                        <div key={i} className="h-24 bg-gray-100 dark:bg-slate-700/50 rounded-xl"></div>
                    ))}
                </div>
            </div>
        );
    }

    if (error) {
        return null; // Don't show if error, just hide section
    }

    if (stations.length === 0) {
        return null; // Don't show if no stations found
    }

    return (
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
                        <MapPin className="w-5 h-5 text-brand-primary" />
                        Monitoring Stations in {city}
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        Real-time readings from {stations.length} locations
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {stations.map((station, index) => {
                    const colorClasses: Record<string, string> = {
                        'good': 'bg-green-500/10 text-green-700 dark:text-green-400 border-green-200 dark:border-green-900',
                        'satisfactory': 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900',
                        'moderate': 'bg-yellow-500/10 text-yellow-700 dark:text-yellow-400 border-yellow-200 dark:border-yellow-900',
                        'poor': 'bg-orange-500/10 text-orange-700 dark:text-orange-400 border-orange-200 dark:border-orange-900',
                        'very-poor': 'bg-red-500/10 text-red-700 dark:text-red-400 border-red-200 dark:border-red-900',
                        'severe': 'bg-rose-500/10 text-rose-800 dark:text-rose-300 border-rose-200 dark:border-rose-900',
                    };

                    const bgClass = colorClasses[station.color] || 'bg-gray-100 border-gray-200';

                    return (
                        <div
                            key={`${station.name}-${index}`}
                            className={`p-4 rounded-xl border ${bgClass} transition-all hover:shadow-md`}
                        >
                            <div className="flex justify-between items-start mb-2">
                                <span className="font-medium text-sm line-clamp-2 min-h-[2.5rem]" title={station.name}>
                                    {station.name.replace(new RegExp(`${city},?`, 'i'), '').trim() || station.name}
                                </span>
                                <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${station.color === 'good' ? 'bg-green-100 text-green-800' : 'bg-white/50'}`}>
                                    {station.status}
                                </span>
                            </div>
                            <div className="flex items-end gap-1">
                                <span className="text-3xl font-bold">
                                    {station.aqi}
                                </span>
                                <span className="text-xs mb-1 opacity-70">AQI</span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
