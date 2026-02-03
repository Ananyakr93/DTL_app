import { useState, useRef, useEffect } from 'react';
import { HelpCircle, X, Leaf, Info, Wind, AlertTriangle, ShieldCheck } from 'lucide-react';
import { useStore } from '../store';

export default function FloatingHelp() {
    const [isOpen, setIsOpen] = useState(false);
    const { settings, currentData } = useStore();
    const isDarkMode = settings.isDarkMode;
    const modalRef = useRef<HTMLDivElement>(null);
    const [hasPulse, setHasPulse] = useState(true);

    // Stop pulsing after first open or if AQI is low
    useEffect(() => {
        if (isOpen) setHasPulse(false);
    }, [isOpen]);

    // Close on outside click
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
        }
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [isOpen]);

    // Auto-pulse if AQI >= 100 on load (if not opened yet)
    useEffect(() => {
        if (currentData?.current.aqi_value && currentData.current.aqi_value >= 100 && !isOpen) {
            setHasPulse(true);
        }
    }, [currentData, isOpen]);

    return (
        <>
            {/* Floating Trigger Button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`fixed bottom-6 right-6 z-50 p-4 rounded-full shadow-2xl transition-all duration-300 transform hover:scale-110 flex items-center justify-center ${isOpen
                    ? 'bg-red-500 hover:bg-red-600 rotate-90 text-white'
                    : isDarkMode ? 'bg-green-600 hover:bg-green-700 text-white' : 'bg-green-600 hover:bg-green-700 text-white'
                    } ${hasPulse && !isOpen ? 'animate-pulse' : ''}`}
                aria-label="Help & Guide"
            >
                {isOpen ? <X className="w-6 h-6" /> : <HelpCircle className="w-6 h-6" />}
            </button>

            {/* Floating Modal Panel */}
            {isOpen && (
                <div
                    ref={modalRef}
                    className={`fixed bottom-24 right-6 left-6 sm:left-auto sm:w-96 z-50 rounded-2xl shadow-2xl border overflow-hidden flex flex-col max-h-[75vh] animate-fade-in-up ${isDarkMode
                        ? 'bg-slate-900 border-slate-700'
                        : 'bg-white border-gray-200'
                        }`}
                >
                    {/* Header */}
                    <div className={`p-4 border-b flex items-center gap-3 ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-green-50/80 border-gray-100'
                        }`}>
                        <div className="p-2 bg-green-100 dark:bg-green-900/50 rounded-full text-green-600 dark:text-green-400">
                            <Leaf className="w-5 h-5" />
                        </div>
                        <div>
                            <h3 className={`font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Air Quality Basics</h3>
                            <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>A Quick Guide for Beginners</p>
                        </div>
                    </div>

                    {/* Content Scroll Area */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-6">

                        {/* Section 1: What is AQI? */}
                        <div className="space-y-2">
                            <h4 className={`text-sm font-bold flex items-center gap-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                                <Info className="w-4 h-4 text-blue-500" />
                                What is AQI?
                            </h4>
                            <p className={`text-sm leading-relaxed ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                The <strong>Air Quality Index (AQI)</strong> is a single number used to report how polluted the air is currently. Think of it like a thermometer for air pollution: the higher the number, the greater the health risk.
                            </p>
                        </div>

                        {/* Section 2: Color Guide */}
                        <div className="space-y-3">
                            <h4 className={`text-sm font-bold flex items-center gap-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                                <ShieldCheck className="w-4 h-4 text-green-500" />
                                Understanding the Colors
                            </h4>
                            <div className="space-y-2 text-xs">
                                <div className="flex items-start gap-3">
                                    <span className="w-3 h-3 rounded-full bg-green-500 mt-1 shrink-0"></span>
                                    <div><span className="font-bold text-green-600 dark:text-green-400">Good (0-50)</span>: Safe for everyone. Open your windows!</div>
                                </div>
                                <div className="flex items-start gap-3">
                                    <span className="w-3 h-3 rounded-full bg-lime-500 mt-1 shrink-0"></span>
                                    <div><span className="font-bold text-lime-600 dark:text-lime-400">Satisfactory (51-100)</span>: Minor breathing discomfort to sensitive people.</div>
                                </div>
                                <div className="flex items-start gap-3">
                                    <span className="w-3 h-3 rounded-full bg-yellow-500 mt-1 shrink-0"></span>
                                    <div><span className="font-bold text-yellow-600 dark:text-yellow-400">Moderate (101-200)</span>: Breathing discomfort with lung/heart disease.</div>
                                </div>
                                <div className="flex items-start gap-3">
                                    <span className="w-3 h-3 rounded-full bg-orange-500 mt-1 shrink-0"></span>
                                    <div><span className="font-bold text-orange-600 dark:text-orange-400">Poor (201-300)</span>: Breathing discomfort on prolonged exposure.</div>
                                </div>
                                <div className="flex items-start gap-3">
                                    <span className="w-3 h-3 rounded-full bg-red-500 mt-1 shrink-0"></span>
                                    <div><span className="font-bold text-red-600 dark:text-red-400">Very Poor (301-400)</span>: Respiratory illness on prolonged exposure.</div>
                                </div>
                                <div className="flex items-start gap-3">
                                    <span className="w-3 h-3 rounded-full bg-purple-600 mt-1 shrink-0"></span>
                                    <div><span className="font-bold text-purple-600 dark:text-purple-400">Severe (400+)</span>: Affects healthy people; serious impact on existing diseases.</div>
                                </div>
                            </div>
                        </div>

                        {/* Section 3: Pollutants */}
                        <div className="space-y-3">
                            <h4 className={`text-sm font-bold flex items-center gap-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                                <Wind className="w-4 h-4 text-gray-500" />
                                Main Pollutants
                            </h4>
                            <div className={`grid grid-cols-2 gap-2 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                <div className={`p-2 rounded border ${isDarkMode ? 'border-slate-800 bg-slate-800/50' : 'border-gray-100 bg-gray-50'}`}>
                                    <strong className="block mb-1">PM2.5</strong>
                                    Tiny particles (smoke/exhaust). Can enter bloodstream.
                                </div>
                                <div className={`p-2 rounded border ${isDarkMode ? 'border-slate-800 bg-slate-800/50' : 'border-gray-100 bg-gray-50'}`}>
                                    <strong className="block mb-1">PM10</strong>
                                    Coarse dust & pollen. Irritates nose/throat.
                                </div>
                                <div className={`p-2 rounded border ${isDarkMode ? 'border-slate-800 bg-slate-800/50' : 'border-gray-100 bg-gray-50'}`}>
                                    <strong className="block mb-1">NO₂</strong>
                                    Traffic/Industrial gas. Lung irritant.
                                </div>
                                <div className={`p-2 rounded border ${isDarkMode ? 'border-slate-800 bg-slate-800/50' : 'border-gray-100 bg-gray-50'}`}>
                                    <strong className="block mb-1">O₃</strong>
                                    Ground-level Ozone. Causes asthma attacks.
                                </div>
                            </div>
                        </div>

                        {/* Section 4: Standards Comparison */}
                        <div className="space-y-3">
                            <h4 className={`text-sm font-bold flex items-center gap-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                                <AlertTriangle className="w-4 h-4 text-orange-500" />
                                Standards Difference
                            </h4>
                            <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                India (CPCB) and Global (WAQI/US-EPA) standards differ slightly. We default to WAQI for global consistency.
                            </p>
                            <div className="overflow-x-auto">
                                <table className={`w-full text-xs text-left ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                    <thead>
                                        <tr className="border-b dark:border-slate-700">
                                            <th className="py-1">Standard</th>
                                            <th className="py-1">Good PM2.5 Limit</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr className="border-b dark:border-slate-800">
                                            <td className="py-1">WAQI (Global)</td>
                                            <td className="py-1">0 - 12 µg/m³</td>
                                        </tr>
                                        <tr>
                                            <td className="py-1">CPCB (India)</td>
                                            <td className="py-1">0 - 30 µg/m³</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    {/* Footer / Dismiss */}
                    <div className={`p-4 border-t ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-gray-50 border-gray-100'}`}>
                        <button
                            onClick={() => setIsOpen(false)}
                            className="w-full py-2 bg-brand-primary text-brand-dark font-bold rounded-lg hover:bg-brand-secondary transition-colors text-sm"
                        >
                            Got it, thanks!
                        </button>
                    </div>
                </div>
            )}
        </>
    );
}
