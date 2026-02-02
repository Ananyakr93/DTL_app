import { useMemo } from 'react';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from 'recharts';
import { TrendingUp, Info } from 'lucide-react';
import { useStore } from '../store';
import { SCENARIO_OPTIONS, getAQIColor, getAQIClass } from '../utils';
import type { Scenario } from '../types';

export default function PredictionChart() {
    const { predictions, scenario, setScenario, settings } = useStore();
    const isDarkMode = settings.isDarkMode;

    // Transform data for Recharts
    const chartData = useMemo(() => {
        return predictions.map((p) => ({
            hour: p.hour,
            aqi: p.aqi,
            aqiLower: p.aqi_lower ?? p.aqi - 10,
            aqiUpper: p.aqi_upper ?? p.aqi + 10,
            status: p.status,
            color: getAQIColor(p.color),
        }));
    }, [predictions]);

    // Get gradient stops based on AQI values
    const gradientStops = useMemo(() => {
        if (chartData.length === 0) return [];
        const maxAqi = Math.max(...chartData.map((d) => d.aqi));
        const minAqi = Math.min(...chartData.map((d) => d.aqi));

        return [
            { offset: '0%', color: getAQIColor(getAQIClass(minAqi)) },
            { offset: '50%', color: getAQIColor(getAQIClass((minAqi + maxAqi) / 2)) },
            { offset: '100%', color: getAQIColor(getAQIClass(maxAqi)) },
        ];
    }, [chartData]);

    // Custom tooltip
    const CustomTooltip = ({
        active,
        payload,
        label,
    }: {
        active?: boolean;
        payload?: Array<{ value: number; payload: (typeof chartData)[0] }>;
        label?: string;
    }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div
                    className={`p-4 rounded-xl shadow-xl border ${isDarkMode ? 'bg-slate-800 border-slate-700 text-white' : 'bg-white border-gray-200 text-gray-900'
                        }`}
                >
                    <p className="font-bold text-lg">{label}</p>
                    <p className="text-2xl font-extrabold" style={{ color: data.color }}>
                        AQI: {data.aqi}
                    </p>
                    <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        Range: {data.aqiLower} - {data.aqiUpper}
                    </p>
                    <p className="text-sm mt-1" style={{ color: data.color }}>
                        {data.status}
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <section
            className={`rounded-2xl overflow-hidden shadow-xl ${isDarkMode ? 'bg-gradient-to-br from-slate-800 to-slate-900' : 'bg-gradient-to-br from-lime-100 to-green-100'
                }`}
        >
            {/* Header */}
            <div className="p-6 pb-4">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                    <h2 className={`text-xl font-bold flex items-center gap-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                        <TrendingUp className="w-6 h-6 text-brand-primary" />
                        AQI Prediction (Next 24 Hours)
                    </h2>

                    {/* Scenario Selector */}
                    <div className="flex items-center gap-3">
                        <label className={`text-sm font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Scenario:</label>
                        <select
                            value={scenario}
                            onChange={(e) => setScenario(e.target.value as Scenario)}
                            className={`px-4 py-2 rounded-xl font-medium transition-all cursor-pointer ${isDarkMode ? 'bg-slate-700 text-white border-slate-600' : 'bg-white text-gray-900 border-gray-200'
                                } border-2 focus:outline-none focus:border-brand-primary`}
                        >
                            {SCENARIO_OPTIONS.map((opt) => (
                                <option key={opt.value} value={opt.value}>
                                    {opt.label}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                {/* Legend */}
                <div className="flex flex-wrap gap-4 mt-4">
                    {[
                        { label: 'Good (0-50)', color: '#22c55e' },
                        { label: 'Satisfactory (51-100)', color: '#84cc16' },
                        { label: 'Moderate (101-200)', color: '#eab308' },
                        { label: 'Poor (201-300)', color: '#f97316' },
                        { label: 'Very Poor (301-400)', color: '#ef4444' },
                        { label: 'Severe (400+)', color: '#9333ea' },
                    ].map(({ label, color }) => (
                        <div key={label} className="flex items-center gap-2 text-xs">
                            <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: color }} />
                            <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>{label}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Chart */}
            <div className={`p-4 mx-4 mb-4 rounded-xl ${isDarkMode ? 'bg-slate-800/50' : 'bg-white/60'}`}>
                <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={chartData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                        <defs>
                            <linearGradient id="aqiGradient" x1="0" y1="0" x2="0" y2="1">
                                {gradientStops.map((stop, i) => (
                                    <stop key={i} offset={stop.offset} stopColor={stop.color} stopOpacity={0.8} />
                                ))}
                            </linearGradient>
                            <linearGradient id="uncertaintyGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#94a3b8" stopOpacity={0.3} />
                                <stop offset="100%" stopColor="#94a3b8" stopOpacity={0.05} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#374151' : '#e5e7eb'} vertical={false} />
                        <XAxis
                            dataKey="hour"
                            tick={{ fill: isDarkMode ? '#9ca3af' : '#6b7280', fontSize: 11 }}
                            tickLine={false}
                            axisLine={{ stroke: isDarkMode ? '#374151' : '#e5e7eb' }}
                            interval={2}
                        />
                        <YAxis
                            tick={{ fill: isDarkMode ? '#9ca3af' : '#6b7280', fontSize: 11 }}
                            tickLine={false}
                            axisLine={false}
                            domain={[0, 'auto']}
                        />
                        <Tooltip content={<CustomTooltip />} />

                        {/* Reference lines for AQI thresholds */}
                        <ReferenceLine y={100} stroke="#84cc16" strokeDasharray="5 5" strokeOpacity={0.5} />
                        <ReferenceLine y={200} stroke="#eab308" strokeDasharray="5 5" strokeOpacity={0.5} />
                        <ReferenceLine y={300} stroke="#f97316" strokeDasharray="5 5" strokeOpacity={0.5} />

                        {/* Uncertainty band (confidence interval) */}
                        <Area type="monotone" dataKey="aqiUpper" stroke="none" fill="url(#uncertaintyGradient)" fillOpacity={1} />
                        <Area
                            type="monotone"
                            dataKey="aqiLower"
                            stroke="none"
                            fill={isDarkMode ? '#1e293b' : '#ffffff'}
                            fillOpacity={1}
                        />

                        {/* Main AQI line */}
                        <Area
                            type="monotone"
                            dataKey="aqi"
                            stroke="url(#aqiGradient)"
                            strokeWidth={3}
                            fill="url(#aqiGradient)"
                            fillOpacity={0.3}
                            dot={{ r: 3, fill: isDarkMode ? '#1e293b' : '#fff', strokeWidth: 2 }}
                            activeDot={{ r: 6, fill: '#a3ff12', stroke: '#fff', strokeWidth: 2 }}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* Footer Note */}
            <div className={`px-6 pb-6 flex items-center gap-2 text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-600'}`}>
                <Info className="w-4 h-4" />
                <span>Shaded area shows prediction uncertainty (±confidence interval)</span>
            </div>
        </section>
    );
}
