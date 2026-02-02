import { useState, useMemo } from 'react';
import { Activity, Clock, HelpCircle } from 'lucide-react';
import { useStore } from '../store';
import { getAQIColor } from '../utils';
import { PieChart, Pie, Cell, ResponsiveContainer, AreaChart, Area, YAxis } from 'recharts';
import AqiInsightsStrip from './AqiInsightsStrip';

export default function AQICard() {
    const { currentData, lastUpdate, refreshCountdown, settings, isLoading, predictions } = useStore();
    const isDarkMode = settings.isDarkMode;

    const [showTooltip, setShowTooltip] = useState(false);

    // Quick win: One-time toast or confetti could go here in useEffect, 
    // but for now we focus on the core insights.

    if (!currentData) {
        return (
            <div className={`rounded-2xl p-6 ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-xl animate-pulse`}>
                <div className="h-32 bg-gray-200 dark:bg-slate-700 rounded-xl" />
            </div>
        );
    }

    const { current } = currentData;
    const aqiColor = getAQIColor(current.aqi_color);

    // Trend Calculation (Simple Slope of last 6 points)
    const trendText = useMemo(() => {
        if (!predictions || predictions.length < 6) return "Stable";
        // Compare avg of last 3 vs prev 3
        const last3 = predictions.slice(-3).reduce((a, b) => a + b.aqi, 0) / 3;
        const prev3 = predictions.slice(-6, -3).reduce((a, b) => a + b.aqi, 0) / 3;
        const diff = last3 - prev3;

        if (diff > 5) return "Air quality worsening";
        if (diff < -5) return "Improving";
        return "Stable";
    }, [predictions]);



    // Gauge Data
    const gaugeData = [
        { name: 'val', value: Math.min(current.aqi_value, 500) },
        { name: 'rest', value: 500 - Math.min(current.aqi_value, 500) }
    ];
    const gaugeColors = [aqiColor, isDarkMode ? '#334155' : '#e2e8f0'];

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
                            {/* Main AQI Display */}
                            <div className="flex items-end gap-4">
                                <div>
                                    <div className="flex items-baseline gap-2">
                                        <span className="text-6xl font-bold tracking-tight">{current.aqi_value}</span>
                                        <span className="text-xl font-medium opacity-80">AQI</span>
                                    </div>
                                    {/* Tooltip for AQI Number */}
                                    <div className="relative group">
                                        <div className="cursor-help flex items-center gap-1 opacity-90 mt-1"
                                            onMouseEnter={() => setShowTooltip(true)}
                                            onMouseLeave={() => setShowTooltip(false)}>
                                            <HelpCircle className="w-3 h-3" />
                                            <span className="text-xs border-b border-dotted border-white/50">What does this mean?</span>
                                        </div>
                                        {showTooltip && (
                                            <div className="absolute bottom-full left-0 mb-2 w-56 p-3 bg-black/90 text-white text-xs rounded-xl shadow-xl z-50 pointer-events-none">
                                                <p className="font-bold mb-1">AQI {current.aqi_value} = {current.aqi_status}</p>
                                                Main driver: {current.dominant_pollutant}. Safe for most, but take precautions if sensitive.
                                            </div>
                                        )}
                                    </div>
                                </div>
                                {/* Sparkline */}
                                <div className="h-16 w-32 pb-2 opacity-90">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={predictions}>
                                            <defs>
                                                <linearGradient id="colorAqi" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#ffffff" stopOpacity={0.8} />
                                                    <stop offset="95%" stopColor="#ffffff" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <YAxis domain={['dataMin', 'dataMax']} hide />
                                            <Area
                                                type="monotone"
                                                dataKey="aqi"
                                                stroke="#ffffff"
                                                strokeWidth={2}
                                                fillOpacity={1}
                                                fill="url(#colorAqi)"
                                            />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>
                        {/* Trend Text */}
                        <div className="flex flex-col items-end justify-center min-w-[80px]">
                            <span className="text-xs text-gray-500 italic opacity-80">
                                {trendText}
                            </span>
                        </div>
                    </div>

                </div>
            </div>



            {/* Compact Gauge (Arc only) */}
            <div className="px-6 pt-4 pb-0 h-16 relative overflow-hidden opacity-50 hover:opacity-100 transition-opacity">
                <ResponsiveContainer width="100%" height={80}>
                    <PieChart>
                        <Pie
                            data={gaugeData}
                            cx="50%"
                            cy="100%"
                            startAngle={180}
                            endAngle={0}
                            innerRadius={60}
                            outerRadius={70}
                            paddingAngle={0}
                            dataKey="value"
                            stroke="none"
                        >
                            {gaugeData.map((_entry, index) => (
                                <Cell key={`cell-${index}`} fill={gaugeColors[index]} />
                            ))}
                        </Pie>
                    </PieChart>
                </ResponsiveContainer>

                {/* Scale Markers */}
                <div className="absolute bottom-0 left-6 text-[10px] opacity-50">0</div>
                <div className="absolute bottom-0 right-6 text-[10px] opacity-50">500</div>
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

            {/* Advanced Insights Strip */}
            <div className="pb-4 border-t border-transparent">
                <AqiInsightsStrip
                    aqi={current.aqi_value}
                    dominantPollutant={current.dominant_pollutant}
                    pm25Value={current.pm2_5}
                    location={currentData.city || 'Unknown'}
                    isDarkMode={isDarkMode}
                />
            </div>
        </div>
    );
}
