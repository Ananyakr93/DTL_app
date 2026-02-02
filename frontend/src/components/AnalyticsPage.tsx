import { useEffect, useState, useMemo } from 'react';
import {
    LineChart,
    Line,
    AreaChart,
    Area,
    PieChart,
    Pie,
    Cell,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from 'recharts';
import { TrendingUp, TrendingDown, Minus, Calendar, BarChart3, Download } from 'lucide-react';
import { useStore } from '../store';
import { fetchHistoricalData } from '../api';
import { calculateStats, exportToCSV, getAQIColor, getAQIClass, formatDate } from '../utils';


const TIME_RANGES = [
    { label: '7 Days', value: 7 },
    { label: '30 Days', value: 30 },
    { label: '90 Days', value: 90 },
];

const POLLUTANT_COLORS: Record<string, string> = {
    pm2_5: '#ef4444',
    pm10: '#f97316',
    no2: '#eab308',
    so2: '#22c55e',
    co: '#3b82f6',
    o3: '#8b5cf6',
};

export default function AnalyticsPage() {
    const { city, historicalData, setHistoricalData, settings } = useStore();
    const isDarkMode = settings.isDarkMode;

    const [timeRange, setTimeRange] = useState(30);
    const [selectedPollutant, setSelectedPollutant] = useState<string>('all');
    const [isLoading, setIsLoading] = useState(true);

    // Fetch historical data
    useEffect(() => {
        const loadData = async () => {
            setIsLoading(true);
            const data = await fetchHistoricalData(city, timeRange);
            setHistoricalData(data);
            setIsLoading(false);
        };
        loadData();
    }, [city, timeRange, setHistoricalData]);

    // Calculate statistics
    const stats = useMemo(() => calculateStats(historicalData), [historicalData]);

    // Prepare pie chart data for pollutant contribution
    const pollutantPieData = useMemo(() => {
        if (historicalData.length === 0) return [];

        const avgPm25 = historicalData.reduce((a, b) => a + b.pm2_5, 0) / historicalData.length;
        const avgPm10 = historicalData.reduce((a, b) => a + b.pm10, 0) / historicalData.length;
        const avgNo2 = historicalData.reduce((a, b) => a + b.no2, 0) / historicalData.length;
        const avgSo2 = historicalData.reduce((a, b) => a + b.so2, 0) / historicalData.length;
        const avgCo = historicalData.reduce((a, b) => a + b.co * 10, 0) / historicalData.length; // Scale CO
        const avgO3 = historicalData.reduce((a, b) => a + b.o3, 0) / historicalData.length;

        return [
            { name: 'PM2.5', value: avgPm25, color: POLLUTANT_COLORS.pm2_5 },
            { name: 'PM10', value: avgPm10, color: POLLUTANT_COLORS.pm10 },
            { name: 'NO₂', value: avgNo2, color: POLLUTANT_COLORS.no2 },
            { name: 'SO₂', value: avgSo2, color: POLLUTANT_COLORS.so2 },
            { name: 'CO', value: avgCo, color: POLLUTANT_COLORS.co },
            { name: 'O₃', value: avgO3, color: POLLUTANT_COLORS.o3 },
        ];
    }, [historicalData]);

    // AQI distribution data
    const aqiDistribution = useMemo(() => {
        const dist = { Good: 0, Satisfactory: 0, Moderate: 0, Poor: 0, 'Very Poor': 0, Severe: 0 };
        historicalData.forEach((d) => {
            const aqi = d.aqi;
            if (aqi <= 50) dist['Good']++;
            else if (aqi <= 100) dist['Satisfactory']++;
            else if (aqi <= 200) dist['Moderate']++;
            else if (aqi <= 300) dist['Poor']++;
            else if (aqi <= 400) dist['Very Poor']++;
            else dist['Severe']++;
        });
        return Object.entries(dist).map(([name, value]) => ({
            name,
            value,
            color: getAQIColor(getAQIClass(name === 'Good' ? 25 : name === 'Satisfactory' ? 75 : name === 'Moderate' ? 150 : name === 'Poor' ? 250 : name === 'Very Poor' ? 350 : 450)),
        }));
    }, [historicalData]);

    const handleExportCSV = () => {
        exportToCSV(historicalData, `aqi-data-${city}-${timeRange}days.csv`);
    };

    if (isLoading) {
        return (
            <div className="flex items-center justify-center min-h-[50vh]">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-brand-primary" />
            </div>
        );
    }

    return (
        <div className="space-y-6 animate-fade-in">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                        <BarChart3 className="inline w-7 h-7 mr-2 text-brand-primary" />
                        Analytics - {city}
                    </h1>
                    <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        Historical air quality trends and pollutant analysis
                    </p>
                </div>

                <div className="flex items-center gap-3">
                    {/* Time Range Selector */}
                    <div className="flex items-center gap-2">
                        <Calendar className={`w-4 h-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`} />
                        <select
                            value={timeRange}
                            onChange={(e) => setTimeRange(Number(e.target.value))}
                            className={`px-3 py-2 rounded-lg text-sm font-medium ${isDarkMode ? 'bg-slate-700 text-white border-slate-600' : 'bg-white text-gray-900 border-gray-200'
                                } border focus:outline-none focus:ring-2 focus:ring-brand-primary`}
                        >
                            {TIME_RANGES.map((range) => (
                                <option key={range.value} value={range.value}>
                                    {range.label}
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* Export Button */}
                    <button
                        onClick={handleExportCSV}
                        className="flex items-center gap-2 px-4 py-2 bg-brand-primary text-brand-dark rounded-lg font-medium hover:bg-opacity-90 transition-colors"
                    >
                        <Download className="w-4 h-4" />
                        Export CSV
                    </button>
                </div>
            </div>

            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className={`p-5 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                    <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Average AQI</p>
                    <p className={`text-3xl font-bold mt-1 ${getAQITextClass(getAQIClass(stats.avgAqi))}`}>{stats.avgAqi}</p>
                </div>

                <div className={`p-5 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                    <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Maximum AQI</p>
                    <p className={`text-3xl font-bold mt-1 text-red-500`}>{stats.maxAqi}</p>
                </div>

                <div className={`p-5 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                    <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Minimum AQI</p>
                    <p className={`text-3xl font-bold mt-1 text-green-500`}>{stats.minAqi}</p>
                </div>

                <div className={`p-5 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                    <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Trend</p>
                    <div className="flex items-center gap-2 mt-1">
                        {stats.trend === 'improving' && <TrendingDown className="w-6 h-6 text-green-500" />}
                        {stats.trend === 'worsening' && <TrendingUp className="w-6 h-6 text-red-500" />}
                        {stats.trend === 'stable' && <Minus className="w-6 h-6 text-yellow-500" />}
                        <span className={`text-xl font-bold capitalize ${stats.trend === 'improving' ? 'text-green-500' : stats.trend === 'worsening' ? 'text-red-500' : 'text-yellow-500'
                            }`}>
                            {stats.trend}
                        </span>
                    </div>
                </div>
            </div>

            {/* AQI Trend Chart */}
            <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                <h3 className={`text-lg font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    AQI Trend Over Time
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={historicalData}>
                        <defs>
                            <linearGradient id="aqiGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#a3ff12" stopOpacity={0.8} />
                                <stop offset="95%" stopColor="#a3ff12" stopOpacity={0.1} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#374151' : '#e5e7eb'} />
                        <XAxis
                            dataKey="date"
                            tick={{ fill: isDarkMode ? '#9ca3af' : '#6b7280', fontSize: 11 }}
                            tickFormatter={(value) => formatDate(value).split(' ').slice(0, 2).join(' ')}
                        />
                        <YAxis tick={{ fill: isDarkMode ? '#9ca3af' : '#6b7280', fontSize: 11 }} />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: isDarkMode ? '#1e293b' : '#fff',
                                border: 'none',
                                borderRadius: '8px',
                                boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
                            }}
                            labelFormatter={(value) => formatDate(value as string)}
                        />
                        <Area type="monotone" dataKey="aqi" stroke="#a3ff12" fill="url(#aqiGradient)" strokeWidth={2} />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* Pollutant Trends & Pie Chart */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Pollutant Trends */}
                <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                    <div className="flex items-center justify-between mb-4">
                        <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                            Pollutant Trends
                        </h3>
                        <select
                            value={selectedPollutant}
                            onChange={(e) => setSelectedPollutant(e.target.value)}
                            className={`px-2 py-1 rounded text-sm ${isDarkMode ? 'bg-slate-700 text-white' : 'bg-gray-100 text-gray-900'
                                }`}
                        >
                            <option value="all">All Pollutants</option>
                            <option value="pm2_5">PM2.5</option>
                            <option value="pm10">PM10</option>
                            <option value="no2">NO₂</option>
                            <option value="so2">SO₂</option>
                            <option value="co">CO</option>
                            <option value="o3">O₃</option>
                        </select>
                    </div>
                    <ResponsiveContainer width="100%" height={250}>
                        <LineChart data={historicalData}>
                            <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#374151' : '#e5e7eb'} />
                            <XAxis
                                dataKey="date"
                                tick={{ fill: isDarkMode ? '#9ca3af' : '#6b7280', fontSize: 10 }}
                                tickFormatter={(value) => formatDate(value).split(' ')[0]}
                            />
                            <YAxis tick={{ fill: isDarkMode ? '#9ca3af' : '#6b7280', fontSize: 10 }} />
                            <Tooltip contentStyle={{ backgroundColor: isDarkMode ? '#1e293b' : '#fff', border: 'none', borderRadius: '8px' }} />
                            <Legend />
                            {(selectedPollutant === 'all' || selectedPollutant === 'pm2_5') && (
                                <Line type="monotone" dataKey="pm2_5" name="PM2.5" stroke={POLLUTANT_COLORS.pm2_5} strokeWidth={2} dot={false} />
                            )}
                            {(selectedPollutant === 'all' || selectedPollutant === 'pm10') && (
                                <Line type="monotone" dataKey="pm10" name="PM10" stroke={POLLUTANT_COLORS.pm10} strokeWidth={2} dot={false} />
                            )}
                            {(selectedPollutant === 'all' || selectedPollutant === 'no2') && (
                                <Line type="monotone" dataKey="no2" name="NO₂" stroke={POLLUTANT_COLORS.no2} strokeWidth={2} dot={false} />
                            )}
                            {(selectedPollutant === 'all' || selectedPollutant === 'so2') && (
                                <Line type="monotone" dataKey="so2" name="SO₂" stroke={POLLUTANT_COLORS.so2} strokeWidth={2} dot={false} />
                            )}
                            {(selectedPollutant === 'all' || selectedPollutant === 'o3') && (
                                <Line type="monotone" dataKey="o3" name="O₃" stroke={POLLUTANT_COLORS.o3} strokeWidth={2} dot={false} />
                            )}
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* Pollutant Contribution Pie */}
                <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                    <h3 className={`text-lg font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                        Average Pollutant Contribution
                    </h3>
                    <ResponsiveContainer width="100%" height={250}>
                        <PieChart>
                            <Pie
                                data={pollutantPieData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={90}
                                paddingAngle={2}
                                dataKey="value"
                                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                labelLine={false}
                            >
                                {pollutantPieData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Pie>
                            <Tooltip />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* AQI Distribution */}
            <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                <h3 className={`text-lg font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    AQI Category Distribution (Last {timeRange} Days)
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
                    {aqiDistribution.map((cat) => (
                        <div key={cat.name} className="text-center">
                            <div
                                className="w-16 h-16 mx-auto rounded-full flex items-center justify-center text-white font-bold text-lg"
                                style={{ backgroundColor: cat.color }}
                            >
                                {cat.value}
                            </div>
                            <p className={`mt-2 text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                                {cat.name}
                            </p>
                            <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                                {historicalData.length > 0 ? ((cat.value / historicalData.length) * 100).toFixed(0) : 0}% of days
                            </p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

function getAQITextClass(colorClass: string): string {
    const classes: Record<string, string> = {
        good: 'text-green-500',
        satisfactory: 'text-lime-500',
        moderate: 'text-yellow-500',
        poor: 'text-orange-500',
        'very-poor': 'text-red-500',
        severe: 'text-purple-600',
    };
    return classes[colorClass] || 'text-gray-500';
}
