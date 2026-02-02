import { Wind, TrendingUp, TrendingDown, Activity, Clock, Database } from 'lucide-react';
import { useStore } from '../store';
import { getAQIColor } from '../utils';

export default function AQICard() {
    const { currentData, lastUpdate, refreshCountdown, settings, isLoading } = useStore();
    const isDarkMode = settings.isDarkMode;

    if (!currentData) {
        return (
            <div className={`rounded-2xl p-6 ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-xl animate-pulse`}>
                <div className="h-32 bg-gray-200 dark:bg-slate-700 rounded-xl" />
            </div>
        );
    }

    const { current } = currentData;
    const aqiColor = getAQIColor(current.aqi_color);

    // Determine trend based on AQI value
    const trend = current.aqi_value > 150 ? 'up' : current.aqi_value < 80 ? 'down' : 'stable';

    return (
        <div
            className={`rounded-2xl overflow-hidden shadow-xl transition-all duration-300 ${isDarkMode ? 'bg-slate-800' : 'bg-white'
                } ${isLoading ? 'opacity-70' : ''}`}
        >
            {/* Header */}
            <div
                className="p-6 text-white relative overflow-hidden"
                style={{ backgroundColor: aqiColor }}
            >
                {/* Background pattern */}
                <div className="absolute inset-0 opacity-10">
                    <div className="absolute -right-10 -top-10 w-40 h-40 rounded-full bg-white" />
                    <div className="absolute -left-10 -bottom-10 w-32 h-32 rounded-full bg-white" />
                </div>

                <div className="relative z-10">
                    <div className="flex items-center justify-between mb-4">
                        <div>
                            <p className="text-white/80 text-sm font-medium flex items-center gap-2">
                                <Activity className="w-4 h-4" />
                                Air Quality Index
                            </p>
                            <h2 className="text-lg font-bold mt-1">{current.station}</h2>
                        </div>
                        <div className="p-2 bg-white/20 rounded-xl">
                            <Wind className="w-6 h-6" />
                        </div>
                    </div>

                    {/* Main AQI Display */}
                    <div className="flex items-end gap-4">
                        <div>
                            <span className="text-6xl font-bold tracking-tight">{current.aqi_value}</span>
                            <span className="text-2xl font-medium ml-2 opacity-80">AQI</span>
                        </div>
                        <div className="mb-2">
                            {trend === 'up' && <TrendingUp className="w-6 h-6 text-white/80" />}
                            {trend === 'down' && <TrendingDown className="w-6 h-6 text-white/80" />}
                        </div>
                    </div>

                    <div className="mt-4 flex items-center justify-between">
                        <span className="px-3 py-1 bg-white/20 rounded-full text-sm font-semibold">
                            {current.aqi_status}
                        </span>
                        <span className="text-white/70 text-sm flex items-center gap-1">
                            <Database className="w-3 h-3" />
                            {current.aqi_source}
                        </span>
                    </div>
                </div>
            </div>

            {/* AQI Scale */}
            <div className="p-4">
                <div className="relative h-3 rounded-full overflow-hidden bg-gray-200 dark:bg-slate-700">
                    <div className="absolute inset-0 flex">
                        <div className="flex-1 bg-green-500" />
                        <div className="flex-1 bg-lime-500" />
                        <div className="flex-1 bg-yellow-500" />
                        <div className="flex-1 bg-orange-500" />
                        <div className="flex-1 bg-red-500" />
                        <div className="flex-1 bg-purple-500" />
                    </div>
                    {/* Marker */}
                    <div
                        className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full border-2 shadow-lg transition-all duration-500"
                        style={{
                            left: `${Math.min((current.aqi_value / 500) * 100, 100)}%`,
                            borderColor: aqiColor,
                        }}
                    />
                </div>
                <div className="flex justify-between text-xs mt-2">
                    <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>0</span>
                    <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>100</span>
                    <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>200</span>
                    <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>300</span>
                    <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>400</span>
                    <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>500</span>
                </div>
            </div>

            {/* Update info */}
            <div className={`px-4 pb-4 flex items-center justify-between text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {lastUpdate ? `Updated ${lastUpdate.toLocaleTimeString()}` : 'Updating...'}
                </div>
                <div className={`px-2 py-1 rounded-full ${isLoading ? 'bg-yellow-100 text-yellow-700' : 'bg-green-100 text-green-700'} dark:bg-opacity-20`}>
                    {isLoading ? 'Refreshing...' : `Next update: ${refreshCountdown}s`}
                </div>
            </div>

            {/* Dominant pollutant */}
            <div className={`px-4 pb-4 border-t ${isDarkMode ? 'border-slate-700' : 'border-gray-100'} pt-3`}>
                <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                    Dominant pollutant: <span className="font-semibold text-brand-primary">{current.dominant_pollutant}</span>
                </p>
            </div>
        </div>
    );
}
