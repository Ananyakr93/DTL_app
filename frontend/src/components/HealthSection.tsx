import { useStore } from '../store';
import { shouldShowHealthWarning } from '../utils';
import { Heart, Shield, AlertTriangle, User, Stethoscope } from 'lucide-react';

export default function HealthSection() {
    const { currentData, settings } = useStore();
    const isDarkMode = settings.isDarkMode;
    const healthConditions = settings.healthConditions;
    const hasSensitiveConditions = healthConditions.some((c) => c !== 'none');

    if (!currentData) return null;

    const { current, health, personal_risk } = currentData;
    const showWarning = shouldShowHealthWarning(current.aqi_value);

    return (
        <section className={`rounded-2xl overflow-hidden shadow-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'}`}>
            <div className="p-6">
                <h2 className={`text-xl font-bold mb-6 flex items-center gap-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    <Heart className={`w-6 h-6 ${showWarning ? 'text-orange-500' : 'text-green-500'}`} />
                    Health Recommendations
                    {hasSensitiveConditions && (
                        <span className="ml-2 px-2 py-0.5 text-xs bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-full">
                            Personalized
                        </span>
                    )}
                </h2>

                {/* Personalized Warning Banner */}
                {hasSensitiveConditions && personal_risk.personalized_warnings && personal_risk.personalized_warnings.length > 0 && (
                    <div className={`mb-6 p-4 rounded-xl border-2 ${isDarkMode ? 'bg-red-900/20 border-red-700' : 'bg-red-50 border-red-200'
                        }`}>
                        <div className="flex items-start gap-3">
                            <User className="w-6 h-6 text-red-500 flex-shrink-0" />
                            <div>
                                <h4 className={`font-bold ${isDarkMode ? 'text-red-400' : 'text-red-700'}`}>
                                    ⚠️ Personalized Alert for Your Health Profile
                                </h4>
                                <ul className={`mt-2 space-y-1 ${isDarkMode ? 'text-red-300' : 'text-red-600'}`}>
                                    {personal_risk.personalized_warnings.map((warning, idx) => (
                                        <li key={idx} className="flex items-start gap-2 text-sm">
                                            <span>•</span>
                                            <span>{warning}</span>
                                        </li>
                                    ))}
                                </ul>
                                <p className={`mt-2 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                                    Based on your health conditions: {healthConditions.filter((c) => c !== 'none').join(', ')}
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* General Recommendations */}
                    <div
                        className={`p-5 rounded-xl border-l-4 ${showWarning
                                ? 'border-l-orange-500 bg-orange-50 dark:bg-orange-900/20'
                                : 'border-l-green-500 bg-green-50 dark:bg-green-900/20'
                            }`}
                    >
                        <div className="flex items-start gap-3">
                            {showWarning ? (
                                <AlertTriangle className="w-6 h-6 text-orange-500 flex-shrink-0 mt-0.5" />
                            ) : (
                                <Shield className="w-6 h-6 text-green-500 flex-shrink-0 mt-0.5" />
                            )}
                            <div>
                                <h3
                                    className={`font-semibold mb-2 ${showWarning
                                            ? 'text-orange-800 dark:text-orange-300'
                                            : 'text-green-800 dark:text-green-300'
                                        }`}
                                >
                                    General Population
                                </h3>
                                <p className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                                    {health.general}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Sensitive Groups */}
                    <div
                        className={`p-5 rounded-xl border-l-4 ${showWarning || hasSensitiveConditions
                                ? 'border-l-red-500 bg-red-50 dark:bg-red-900/20'
                                : 'border-l-blue-500 bg-blue-50 dark:bg-blue-900/20'
                            }`}
                    >
                        <div className="flex items-start gap-3">
                            <Stethoscope
                                className={`w-6 h-6 flex-shrink-0 mt-0.5 ${showWarning || hasSensitiveConditions ? 'text-red-500' : 'text-blue-500'
                                    }`}
                            />
                            <div>
                                <h3
                                    className={`font-semibold mb-2 ${showWarning || hasSensitiveConditions
                                            ? 'text-red-800 dark:text-red-300'
                                            : 'text-blue-800 dark:text-blue-300'
                                        }`}
                                >
                                    Sensitive Groups
                                    {hasSensitiveConditions && <span className="ml-2 text-xs">(Applies to you)</span>}
                                </h3>
                                <p className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                                    {health.sensitive}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Outdoor & Mask Advice */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                    <div className={`p-4 rounded-xl ${isDarkMode ? 'bg-slate-700' : 'bg-gray-50'}`}>
                        <h4 className={`font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            🌳 Outdoor Activity
                        </h4>
                        <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                            {health.outdoor_advice}
                        </p>
                    </div>
                    <div className={`p-4 rounded-xl ${isDarkMode ? 'bg-slate-700' : 'bg-gray-50'}`}>
                        <h4 className={`font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            😷 Mask Recommendation
                        </h4>
                        <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                            {health.mask_advice}
                        </p>
                    </div>
                </div>

                {/* Specific Advice */}
                {health.specific_advice && health.specific_advice.length > 0 && (
                    <div className="mt-6">
                        <h3 className={`font-semibold mb-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            Pollutant-Specific Advice
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {health.specific_advice.map((advice, index) => (
                                <div
                                    key={index}
                                    className={`p-4 rounded-lg ${isDarkMode ? 'bg-slate-700' : 'bg-gray-50'}`}
                                >
                                    <div className="flex items-center gap-2 mb-2">
                                        <span className="font-medium text-brand-primary">{advice.pollutant}</span>
                                    </div>
                                    <p className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                                        {advice.message}
                                    </p>
                                    <p className={`text-xs mt-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                        💡 {advice.action}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </section>
    );
}
