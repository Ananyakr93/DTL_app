import { AlertTriangle, X, ShieldAlert, Stethoscope, Heart } from 'lucide-react';
import { useState } from 'react';
import type { PersonalRisk } from '../types';
import { useStore } from '../store';

interface HealthAlertProps {
    risk: PersonalRisk;
}

export default function HealthAlert({ risk }: HealthAlertProps) {
    const [isDismissed, setIsDismissed] = useState(false);
    const { settings } = useStore();
    const isDarkMode = settings.isDarkMode;
    const hasSensitiveConditions = settings.healthConditions.some((c) => c !== 'none');

    if (isDismissed) return null;

    const riskColors = {
        Low: { bg: 'bg-blue-500', border: 'border-blue-500', text: 'text-blue-500', bgLight: 'bg-blue-50 dark:bg-blue-900/20' },
        Moderate: { bg: 'bg-yellow-500', border: 'border-yellow-500', text: 'text-yellow-500', bgLight: 'bg-yellow-50 dark:bg-yellow-900/20' },
        High: { bg: 'bg-orange-500', border: 'border-orange-500', text: 'text-orange-500', bgLight: 'bg-orange-50 dark:bg-orange-900/20' },
        Severe: { bg: 'bg-red-500', border: 'border-red-500', text: 'text-red-500', bgLight: 'bg-red-50 dark:bg-red-900/20' },
    };

    const colors = riskColors[risk.risk_level] || riskColors.Moderate;

    return (
        <div
            className={`rounded-2xl overflow-hidden shadow-xl animate-fade-in border-l-4 ${colors.border} ${isDarkMode ? 'bg-slate-800' : 'bg-white'
                }`}
        >
            <div className="p-6">
                <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-4 flex-1">
                        <div className={`p-3 rounded-xl ${colors.bgLight}`}>
                            {risk.risk_level === 'Severe' ? (
                                <ShieldAlert className={`w-8 h-8 ${colors.text}`} />
                            ) : (
                                <AlertTriangle className={`w-8 h-8 ${colors.text}`} />
                            )}
                        </div>

                        <div className="flex-1">
                            <div className="flex items-center gap-2 flex-wrap">
                                <h3 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                    {risk.alert_title}
                                </h3>
                                {hasSensitiveConditions && (
                                    <span className="px-2 py-0.5 text-xs bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-full flex items-center gap-1">
                                        <Heart className="w-3 h-3" />
                                        Personalized Alert
                                    </span>
                                )}
                            </div>
                            <p className={`mt-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                                {risk.alert_message}
                            </p>

                            {/* Personalized Warnings */}
                            {risk.personalized_warnings && risk.personalized_warnings.length > 0 && (
                                <div className={`mt-4 p-3 rounded-lg ${isDarkMode ? 'bg-red-900/20' : 'bg-red-50'}`}>
                                    <p className={`font-semibold text-sm ${isDarkMode ? 'text-red-400' : 'text-red-700'}`}>
                                        ⚠️ Based on your health profile:
                                    </p>
                                    <ul className="mt-1 space-y-1">
                                        {risk.personalized_warnings.map((warning, idx) => (
                                            <li key={idx} className={`text-sm flex items-start gap-2 ${isDarkMode ? 'text-red-300' : 'text-red-600'}`}>
                                                <span>•</span>
                                                <span>{warning}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {/* Tips */}
                            {risk.tips && risk.tips.length > 0 && (
                                <div className="mt-4">
                                    <p className={`text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                        Recommended Actions:
                                    </p>
                                    <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                        {risk.tips.map((tip, index) => (
                                            <li key={index} className="flex items-start gap-2">
                                                <Stethoscope className={`w-4 h-4 mt-0.5 flex-shrink-0 ${colors.text}`} />
                                                <span className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                                                    {tip}
                                                </span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {/* Health conditions reminder */}
                            {hasSensitiveConditions && (
                                <div className={`mt-4 pt-3 border-t ${isDarkMode ? 'border-slate-700' : 'border-gray-200'}`}>
                                    <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                                        Your health conditions: {settings.healthConditions.filter((c) => c !== 'none').join(', ')}
                                        {' '}<a href="#" onClick={(e) => { e.preventDefault(); useStore.getState().setActivePage('settings'); }} className="text-brand-primary hover:underline">Edit in Settings</a>
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Dismiss Button */}
                    <button
                        onClick={() => setIsDismissed(true)}
                        className={`p-2 rounded-lg transition-colors ${isDarkMode ? 'hover:bg-slate-700 text-gray-400' : 'hover:bg-gray-100 text-gray-500'
                            }`}
                        aria-label="Dismiss alert"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>
            </div>
        </div>
    );
}
