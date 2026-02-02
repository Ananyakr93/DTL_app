import { useState } from 'react';
import { Settings, Moon, Sun, MapPin, Bell, RefreshCw, Save, RotateCcw, Heart, Stethoscope } from 'lucide-react';
import { useStore } from '../store';
import { INDIAN_CITIES } from '../data/cities';
import type { HealthCondition } from '../types';

// Health condition options
const HEALTH_CONDITIONS: Array<{
    id: HealthCondition;
    label: string;
    description: string;
    icon: string;
    color: string;
}> = [
        { id: 'asthma', label: 'Asthma', description: 'Chronic respiratory condition', icon: '🫁', color: 'blue' },
        { id: 'respiratory', label: 'Respiratory Issues', description: 'COPD, bronchitis, lung conditions', icon: '🌬️', color: 'cyan' },
        { id: 'heart_disease', label: 'Heart Disease', description: 'Cardiovascular conditions', icon: '❤️', color: 'red' },
        { id: 'pregnant', label: 'Pregnant', description: 'Expecting mother', icon: '🤰', color: 'pink' },
        { id: 'elderly', label: 'Elderly (60+)', description: 'Senior citizen', icon: '👴', color: 'purple' },
        { id: 'children', label: 'Children (under 12)', description: 'Young children in household', icon: '👶', color: 'orange' },
        { id: 'none', label: 'None / Healthy Adult', description: 'No special health concerns', icon: '💪', color: 'green' },
    ];

export default function SettingsPage() {
    const { settings, updateSettings, city, setCity } = useStore();
    const [localSettings, setLocalSettings] = useState(settings);
    const [isSaved, setIsSaved] = useState(false);

    const handleSave = () => {
        updateSettings(localSettings);
        if (localSettings.defaultCity !== city) {
            setCity(localSettings.defaultCity);
        }
        setIsSaved(true);
        setTimeout(() => setIsSaved(false), 2000);
    };

    const handleReset = () => {
        const defaults = {
            isDarkMode: false,
            defaultCity: 'Bangalore',
            units: 'metric' as const,
            alertThreshold: 100,
            enableNotifications: true,
            refreshInterval: 60,
            healthConditions: ['none'] as HealthCondition[],
        };
        setLocalSettings(defaults);
        updateSettings(defaults);
    };

    const toggleHealthCondition = (condition: HealthCondition) => {
        if (condition === 'none') {
            // If selecting "none", clear all other conditions
            setLocalSettings((prev) => ({
                ...prev,
                healthConditions: ['none'],
            }));
        } else {
            // Toggle the condition
            setLocalSettings((prev) => {
                const conditions = prev.healthConditions.filter((c) => c !== 'none');
                const hasCondition = conditions.includes(condition);

                if (hasCondition) {
                    const newConditions = conditions.filter((c) => c !== condition);
                    return {
                        ...prev,
                        healthConditions: newConditions.length === 0 ? ['none'] : newConditions,
                    };
                } else {
                    return {
                        ...prev,
                        healthConditions: [...conditions, condition],
                    };
                }
            });
        }
    };

    const isConditionSelected = (condition: HealthCondition) => {
        return localSettings.healthConditions.includes(condition);
    };

    const isDarkMode = localSettings.isDarkMode;

    return (
        <div className="space-y-6 animate-fade-in max-w-4xl">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                        <Settings className="inline w-7 h-7 mr-2 text-brand-primary" />
                        Settings
                    </h1>
                    <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        Customize your dashboard and health profile
                    </p>
                </div>

                <div className="flex gap-3">
                    <button
                        onClick={handleReset}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${isDarkMode ? 'bg-slate-700 text-gray-300 hover:bg-slate-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                            }`}
                    >
                        <RotateCcw className="w-4 h-4" />
                        Reset
                    </button>
                    <button
                        onClick={handleSave}
                        className="flex items-center gap-2 px-4 py-2 bg-brand-primary text-brand-dark rounded-lg font-medium hover:bg-opacity-90 transition-colors"
                    >
                        <Save className="w-4 h-4" />
                        {isSaved ? 'Saved!' : 'Save Changes'}
                    </button>
                </div>
            </div>

            {/* Personal Health Profile - PROMINENT SECTION */}
            <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-gradient-to-br from-slate-800 to-slate-900' : 'bg-gradient-to-br from-red-50 to-pink-50'} shadow-xl border-2 ${isDarkMode ? 'border-red-900/30' : 'border-red-200'}`}>
                <h3 className={`text-xl font-bold mb-2 flex items-center gap-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    <div className="p-2 rounded-xl bg-red-500/20">
                        <Heart className="w-6 h-6 text-red-500" />
                    </div>
                    My Health Profile
                </h3>
                <p className={`mb-6 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Select your health conditions to receive personalized air quality recommendations.
                    This helps us provide stricter advice when needed.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {HEALTH_CONDITIONS.map((condition) => {
                        const isSelected = isConditionSelected(condition.id);
                        const colorClasses: Record<string, string> = {
                            blue: isSelected ? 'bg-blue-500 border-blue-500' : 'border-blue-300 hover:border-blue-400',
                            cyan: isSelected ? 'bg-cyan-500 border-cyan-500' : 'border-cyan-300 hover:border-cyan-400',
                            red: isSelected ? 'bg-red-500 border-red-500' : 'border-red-300 hover:border-red-400',
                            pink: isSelected ? 'bg-pink-500 border-pink-500' : 'border-pink-300 hover:border-pink-400',
                            purple: isSelected ? 'bg-purple-500 border-purple-500' : 'border-purple-300 hover:border-purple-400',
                            orange: isSelected ? 'bg-orange-500 border-orange-500' : 'border-orange-300 hover:border-orange-400',
                            green: isSelected ? 'bg-green-500 border-green-500' : 'border-green-300 hover:border-green-400',
                        };

                        return (
                            <button
                                key={condition.id}
                                onClick={() => toggleHealthCondition(condition.id)}
                                className={`flex items-start gap-4 p-4 rounded-xl border-2 transition-all ${isSelected
                                    ? `${colorClasses[condition.color]} text-white`
                                    : `${isDarkMode ? 'bg-slate-800 border-slate-600 hover:bg-slate-700' : 'bg-white'} ${colorClasses[condition.color]}`
                                    }`}
                            >
                                <span className="text-2xl">{condition.icon}</span>
                                <div className="text-left flex-1">
                                    <p className={`font-semibold ${isSelected ? 'text-white' : isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                        {condition.label}
                                    </p>
                                    <p className={`text-sm ${isSelected ? 'text-white/80' : isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                        {condition.description}
                                    </p>
                                </div>
                                {isSelected && (
                                    <span className="w-6 h-6 bg-white rounded-full flex items-center justify-center">
                                        <span className="text-green-600">✓</span>
                                    </span>
                                )}
                            </button>
                        );
                    })}
                </div>

                {/* Selected conditions summary */}
                <div className={`mt-4 p-4 rounded-lg ${isDarkMode ? 'bg-slate-800' : 'bg-white/80'}`}>
                    <p className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                        <Stethoscope className="w-4 h-4 inline mr-2" />
                        Active health conditions:
                    </p>
                    <div className="flex flex-wrap gap-2 mt-2">
                        {localSettings.healthConditions.map((cond) => {
                            const condInfo = HEALTH_CONDITIONS.find((c) => c.id === cond);
                            return (
                                <span
                                    key={cond}
                                    className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium ${isDarkMode ? 'bg-slate-700 text-gray-300' : 'bg-gray-100 text-gray-700'
                                        }`}
                                >
                                    {condInfo?.icon} {condInfo?.label}
                                </span>
                            );
                        })}
                    </div>
                </div>
            </div>

            {/* Appearance */}
            <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                <h3 className={`text-lg font-semibold mb-6 flex items-center gap-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    {isDarkMode ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
                    Appearance
                </h3>

                <div className="space-y-6">
                    {/* Dark Mode Toggle */}
                    <div className="flex items-center justify-between">
                        <div>
                            <p className={`font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Dark Mode</p>
                            <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                Switch between light and dark theme
                            </p>
                        </div>
                        <button
                            onClick={() => setLocalSettings({ ...localSettings, isDarkMode: !localSettings.isDarkMode })}
                            className={`relative w-14 h-8 rounded-full transition-colors ${localSettings.isDarkMode ? 'bg-brand-primary' : 'bg-gray-300'
                                }`}
                        >
                            <span
                                className={`absolute top-1 w-6 h-6 rounded-full bg-white shadow transition-transform ${localSettings.isDarkMode ? 'left-7' : 'left-1'
                                    }`}
                            />
                        </button>
                    </div>
                </div>
            </div>

            {/* Location Settings */}
            <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                <h3 className={`text-lg font-semibold mb-6 flex items-center gap-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    <MapPin className="w-5 h-5" />
                    Location
                </h3>

                <div className="space-y-6">
                    {/* Default City */}
                    <div>
                        <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            Default City
                        </label>
                        <select
                            value={localSettings.defaultCity}
                            onChange={(e) => setLocalSettings({ ...localSettings, defaultCity: e.target.value })}
                            className={`w-full px-4 py-2 rounded-lg border ${isDarkMode ? 'bg-slate-700 text-white border-slate-600' : 'bg-white text-gray-900 border-gray-200'
                                } focus:outline-none focus:ring-2 focus:ring-brand-primary`}
                        >
                            {INDIAN_CITIES.map((c) => (
                                <option key={c.name} value={c.name}>
                                    {c.name}, {c.state}
                                </option>
                            ))}
                        </select>
                        <p className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                            The dashboard will load this city by default
                        </p>
                    </div>

                    {/* Units */}
                    <div>
                        <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            Measurement Units
                        </label>
                        <div className="flex gap-4">
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="radio"
                                    name="units"
                                    value="metric"
                                    checked={localSettings.units === 'metric'}
                                    onChange={() => setLocalSettings({ ...localSettings, units: 'metric' })}
                                    className="w-4 h-4 text-brand-primary"
                                />
                                <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>Metric (µg/m³)</span>
                            </label>
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="radio"
                                    name="units"
                                    value="imperial"
                                    checked={localSettings.units === 'imperial'}
                                    onChange={() => setLocalSettings({ ...localSettings, units: 'imperial' })}
                                    className="w-4 h-4 text-brand-primary"
                                />
                                <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>Imperial (ppm)</span>
                            </label>
                        </div>
                    </div>
                </div>
            </div>

            {/* Notifications */}
            <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                <h3 className={`text-lg font-semibold mb-6 flex items-center gap-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    <Bell className="w-5 h-5" />
                    Notifications
                </h3>

                <div className="space-y-6">
                    {/* Enable Notifications */}
                    <div className="flex items-center justify-between">
                        <div>
                            <p className={`font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Enable Alerts</p>
                            <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                Show health alerts when AQI exceeds threshold
                            </p>
                        </div>
                        <button
                            onClick={() => setLocalSettings({ ...localSettings, enableNotifications: !localSettings.enableNotifications })}
                            className={`relative w-14 h-8 rounded-full transition-colors ${localSettings.enableNotifications ? 'bg-brand-primary' : 'bg-gray-300'
                                }`}
                        >
                            <span
                                className={`absolute top-1 w-6 h-6 rounded-full bg-white shadow transition-transform ${localSettings.enableNotifications ? 'left-7' : 'left-1'
                                    }`}
                            />
                        </button>
                    </div>

                    {/* Alert Threshold */}
                    <div>
                        <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            Alert Threshold (AQI)
                        </label>
                        <div className="flex items-center gap-4">
                            <input
                                type="range"
                                min="50"
                                max="300"
                                step="10"
                                value={localSettings.alertThreshold}
                                onChange={(e) => setLocalSettings({ ...localSettings, alertThreshold: Number(e.target.value) })}
                                className="flex-1 accent-brand-primary"
                            />
                            <span className={`w-12 text-center font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                {localSettings.alertThreshold}
                            </span>
                        </div>
                        <div className="flex justify-between text-xs mt-1">
                            <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>50 (Good)</span>
                            <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>300 (Poor)</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Data Refresh */}
            <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                <h3 className={`text-lg font-semibold mb-6 flex items-center gap-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    <RefreshCw className="w-5 h-5" />
                    Data & Refresh
                </h3>

                <div className="space-y-6">
                    {/* Refresh Interval */}
                    <div>
                        <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            Auto-Refresh Interval
                        </label>
                        <select
                            value={localSettings.refreshInterval}
                            onChange={(e) => setLocalSettings({ ...localSettings, refreshInterval: Number(e.target.value) })}
                            className={`w-full px-4 py-2 rounded-lg border ${isDarkMode ? 'bg-slate-700 text-white border-slate-600' : 'bg-white text-gray-900 border-gray-200'
                                } focus:outline-none focus:ring-2 focus:ring-brand-primary`}
                        >
                            <option value={30}>Every 30 seconds</option>
                            <option value={60}>Every 1 minute</option>
                            <option value={120}>Every 2 minutes</option>
                            <option value={300}>Every 5 minutes</option>
                            <option value={600}>Every 10 minutes</option>
                        </select>
                    </div>
                </div>
            </div>

            {/* About */}
            <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                <h3 className={`text-lg font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    About AeroClean
                </h3>
                <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    <p>Version 2.1.0</p>
                    <p className="mt-2">
                        AeroClean is a real-time air quality monitoring dashboard with personalized health recommendations.
                    </p>
                    <p className="mt-2">
                        Data Sources: World Air Quality Index (WAQI), Central Pollution Control Board (CPCB)
                    </p>
                    <div className="mt-4 pt-4 border-t border-gray-200 dark:border-slate-700">
                        <p className="text-xs">
                            © 2024 AeroClean. For informational purposes only. Consult local authorities and healthcare providers for medical decisions.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
