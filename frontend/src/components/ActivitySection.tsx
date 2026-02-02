import { useStore } from '../store';
import { Activity } from 'lucide-react';

export default function ActivitySection() {
    const { currentData, settings } = useStore();
    const isDarkMode = settings.isDarkMode;

    if (!currentData || !currentData.activities) return null;

    const { activities, current } = currentData;
    const isGoodAir = current.aqi_value <= 100;

    return (
        <section className={`rounded-2xl p-6 ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-xl`}>
            <h2 className={`text-xl font-bold mb-6 flex items-center gap-3 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                <Activity className={`w-6 h-6 ${isGoodAir ? 'text-green-500' : 'text-orange-500'}`} />
                Activity Recommendations
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {activities.map((activity, index) => {
                    const isAllowed = activity.startsWith('✅');
                    const isWarning = activity.startsWith('⚠️');
                    const isDanger = activity.startsWith('❌') || activity.startsWith('🚨');

                    let bgColor, borderColor, textColor;
                    if (isAllowed) {
                        bgColor = isDarkMode ? 'bg-green-900/20' : 'bg-green-50';
                        borderColor = 'border-l-green-500';
                        textColor = isDarkMode ? 'text-green-400' : 'text-green-800';
                    } else if (isWarning) {
                        bgColor = isDarkMode ? 'bg-yellow-900/20' : 'bg-yellow-50';
                        borderColor = 'border-l-yellow-500';
                        textColor = isDarkMode ? 'text-yellow-400' : 'text-yellow-800';
                    } else if (isDanger) {
                        bgColor = isDarkMode ? 'bg-red-900/20' : 'bg-red-50';
                        borderColor = 'border-l-red-500';
                        textColor = isDarkMode ? 'text-red-400' : 'text-red-800';
                    } else {
                        bgColor = isDarkMode ? 'bg-slate-700' : 'bg-gray-50';
                        borderColor = 'border-l-gray-400';
                        textColor = isDarkMode ? 'text-gray-300' : 'text-gray-700';
                    }

                    return (
                        <div
                            key={index}
                            className={`flex items-center gap-3 p-4 rounded-xl border-l-4 transition-all card-hover ${bgColor} ${borderColor}`}
                        >
                            <span className={`text-sm font-medium ${textColor}`}>{activity}</span>
                        </div>
                    );
                })}
            </div>
        </section>
    );
}
