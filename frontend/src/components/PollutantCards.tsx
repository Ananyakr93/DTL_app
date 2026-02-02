import { Droplets, Cloud, Factory, Flame, Wind, Sun, Thermometer, Waves } from 'lucide-react';
import { useStore } from '../store';

interface PollutantInfo {
    id: string;
    name: string;
    fullName: string;
    icon: React.ReactNode;
    unit: string;
    safeLimit: number;
    color: string;
    bgColor: string;
}

const pollutantInfo: PollutantInfo[] = [
    {
        id: 'pm2_5',
        name: 'PM2.5',
        fullName: 'Fine Particulate Matter',
        icon: <Droplets className="w-5 h-5" />,
        unit: 'µg/m³',
        safeLimit: 35,
        color: 'text-purple-500',
        bgColor: 'bg-purple-500/20',
    },
    {
        id: 'pm10',
        name: 'PM10',
        fullName: 'Coarse Particulate Matter',
        icon: <Cloud className="w-5 h-5" />,
        unit: 'µg/m³',
        safeLimit: 100,
        color: 'text-blue-500',
        bgColor: 'bg-blue-500/20',
    },
    {
        id: 'no2',
        name: 'NO₂',
        fullName: 'Nitrogen Dioxide',
        icon: <Factory className="w-5 h-5" />,
        unit: 'µg/m³',
        safeLimit: 80,
        color: 'text-orange-500',
        bgColor: 'bg-orange-500/20',
    },
    {
        id: 'so2',
        name: 'SO₂',
        fullName: 'Sulfur Dioxide',
        icon: <Flame className="w-5 h-5" />,
        unit: 'µg/m³',
        safeLimit: 80,
        color: 'text-yellow-500',
        bgColor: 'bg-yellow-500/20',
    },
    {
        id: 'co',
        name: 'CO',
        fullName: 'Carbon Monoxide',
        icon: <Wind className="w-5 h-5" />,
        unit: 'mg/m³',
        safeLimit: 4,
        color: 'text-red-500',
        bgColor: 'bg-red-500/20',
    },
    {
        id: 'o3',
        name: 'O₃',
        fullName: 'Ozone',
        icon: <Sun className="w-5 h-5" />,
        unit: 'µg/m³',
        safeLimit: 100,
        color: 'text-cyan-500',
        bgColor: 'bg-cyan-500/20',
    },
];

export default function PollutantCards() {
    const { currentData, settings, isLoading } = useStore();
    const isDarkMode = settings.isDarkMode;

    if (!currentData) {
        return (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {[...Array(6)].map((_, i) => (
                    <div key={i} className={`rounded-xl p-4 ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg animate-pulse`}>
                        <div className="h-20 bg-gray-200 dark:bg-slate-700 rounded-lg" />
                    </div>
                ))}
            </div>
        );
    }

    const { current } = currentData;

    const getPollutantValue = (id: string): number | null => {
        switch (id) {
            case 'pm2_5': return current.pm2_5;
            case 'pm10': return current.pm10;
            case 'no2': return current.no2;
            case 'so2': return current.so2;
            case 'co': return current.co;
            case 'o3': return current.o3;
            default: return null;
        }
    };

    const getStatusColor = (value: number | null, safeLimit: number): string => {
        if (value === null) return 'text-gray-400';
        if (value <= safeLimit * 0.5) return 'text-green-500';
        if (value <= safeLimit) return 'text-yellow-500';
        if (value <= safeLimit * 1.5) return 'text-orange-500';
        return 'text-red-500';
    };

    const getStatus = (value: number | null, safeLimit: number): string => {
        if (value === null) return 'N/A';
        if (value <= safeLimit * 0.5) return 'Good';
        if (value <= safeLimit) return 'Moderate';
        if (value <= safeLimit * 1.5) return 'Poor';
        return 'Severe';
    };

    return (
        <div className={`${isLoading ? 'opacity-70' : ''}`}>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {pollutantInfo.map((pollutant) => {
                    const value = getPollutantValue(pollutant.id);
                    const statusColor = getStatusColor(value, pollutant.safeLimit);
                    const status = getStatus(value, pollutant.safeLimit);
                    const percentage = value !== null ? Math.min((value / pollutant.safeLimit) * 100, 150) : 0;

                    return (
                        <div
                            key={pollutant.id}
                            className={`rounded-xl p-4 transition-all duration-300 ${isDarkMode ? 'bg-slate-800 hover:bg-slate-750' : 'bg-white hover:shadow-lg'
                                } shadow-md group`}
                        >
                            <div className="flex items-center justify-between mb-3">
                                <div className={`p-2 rounded-lg ${pollutant.bgColor}`}>
                                    <span className={pollutant.color}>{pollutant.icon}</span>
                                </div>
                                <span className={`text-xs font-medium px-2 py-1 rounded-full ${status === 'Good' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                                    status === 'Moderate' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                                        status === 'Poor' ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400' :
                                            status === 'Severe' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                                                'bg-gray-100 text-gray-500 dark:bg-gray-800 dark:text-gray-400'
                                    }`}>
                                    {status}
                                </span>
                            </div>

                            <div className="mb-2">
                                <h3 className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                    {pollutant.name}
                                </h3>
                                <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                                    {pollutant.fullName}
                                </p>
                            </div>

                            <div className="flex items-baseline gap-1 mb-3">
                                <span className={`text-2xl font-bold ${statusColor}`}>
                                    {value !== null ? (typeof value === 'number' ? value.toFixed(1) : value) : '--'}
                                </span>
                                <span className={`text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                                    {pollutant.unit}
                                </span>
                            </div>

                            {/* Progress bar */}
                            <div className="relative h-2 rounded-full bg-gray-200 dark:bg-slate-700 overflow-hidden">
                                <div
                                    className={`absolute inset-y-0 left-0 rounded-full transition-all duration-500 ${percentage <= 50 ? 'bg-green-500' :
                                        percentage <= 100 ? 'bg-yellow-500' :
                                            percentage <= 150 ? 'bg-orange-500' :
                                                'bg-red-500'
                                        }`}
                                    style={{ width: `${Math.min(percentage, 100)}%` }}
                                />
                            </div>

                            <div className="mt-2 flex justify-between text-xs">
                                <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>0</span>
                                <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>
                                    Safe: {pollutant.safeLimit}
                                </span>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Weather info if available */}
            {(current.temperature !== null || current.humidity !== null || current.wind !== null) && (
                <div className={`mt-4 grid grid-cols-3 gap-4`}>
                    {current.temperature !== null && (
                        <div className={`p-4 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-md`}>
                            <div className="flex items-center gap-2">
                                <Thermometer className="w-4 h-4 text-orange-500" />
                                <span className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Temperature</span>
                            </div>
                            <p className={`text-xl font-bold mt-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                {current.temperature?.toFixed(1) ?? '--'}°C
                            </p>
                        </div>
                    )}
                    {current.humidity !== null && (
                        <div className={`p-4 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-md`}>
                            <div className="flex items-center gap-2">
                                <Droplets className="w-4 h-4 text-blue-500" />
                                <span className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Humidity</span>
                            </div>
                            <p className={`text-xl font-bold mt-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                {current.humidity?.toFixed(0) ?? '--'}%
                            </p>
                        </div>
                    )}
                    {current.wind !== null && (
                        <div className={`p-4 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-md`}>
                            <div className="flex items-center gap-2">
                                <Waves className="w-4 h-4 text-cyan-500" />
                                <span className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Wind Speed</span>
                            </div>
                            <p className={`text-xl font-bold mt-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                {current.wind?.toFixed(1) ?? '--'} m/s
                            </p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
