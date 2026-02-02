import { Loader2 } from 'lucide-react';
import { useStore } from '../store';

export default function LoadingState() {
    const { settings } = useStore();
    const isDarkMode = settings.isDarkMode;

    return (
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-6">
            {/* Animated Spinner */}
            <div className="relative">
                <div className={`w-20 h-20 border-4 ${isDarkMode ? 'border-slate-700' : 'border-gray-200'} rounded-full`}>
                    <div className="absolute inset-0 border-4 border-brand-primary border-t-transparent rounded-full animate-spin" />
                </div>
                <Loader2 className="absolute inset-0 m-auto w-8 h-8 text-brand-primary animate-pulse" />
            </div>

            {/* Loading Text */}
            <div className="text-center space-y-2">
                <h3 className={`text-xl font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    Loading Air Quality Data
                </h3>
                <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Fetching real-time data from monitoring stations...
                </p>
            </div>

            {/* Skeleton Cards */}
            <div className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-4 gap-4 px-4">
                {[1, 2, 3, 4].map((i) => (
                    <div
                        key={i}
                        className={`rounded-xl p-6 ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}
                    >
                        <div className={`h-4 w-20 rounded mb-4 ${isDarkMode ? 'bg-slate-700' : 'bg-gray-200'} animate-pulse`} />
                        <div className={`h-10 w-24 rounded mb-2 ${isDarkMode ? 'bg-slate-700' : 'bg-gray-200'} animate-pulse`} />
                        <div className={`h-6 w-16 rounded-full ${isDarkMode ? 'bg-slate-700' : 'bg-gray-200'} animate-pulse`} />
                    </div>
                ))}
            </div>
        </div>
    );
}
