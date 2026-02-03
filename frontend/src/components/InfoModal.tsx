
import { X } from 'lucide-react';

interface InfoModalProps {
    isOpen: boolean;
    onClose: () => void;
    isDarkMode: boolean;
}

export default function InfoModal({ isOpen, onClose, isDarkMode }: InfoModalProps) {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
            <div
                className={`w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl shadow-2xl ${isDarkMode ? 'bg-slate-900 border border-slate-700 text-white' : 'bg-white text-gray-900'
                    } relative`}
            >
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full hover:bg-gray-200 dark:hover:bg-slate-800 transition-colors"
                >
                    <X className="w-5 h-5" />
                </button>

                <div className="p-6 md:p-8 space-y-6">
                    <header>
                        <h2 className="text-2xl font-bold bg-gradient-to-r from-brand-primary to-green-400 bg-clip-text text-transparent">
                            Understanding Air Quality
                        </h2>
                        <p className={`mt-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                            A quick guide to AQI, pollutants, and your health.
                        </p>
                    </header>

                    <section className="space-y-4">
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                            🧪 What is AQI?
                        </h3>
                        <p className={`text-sm leading-relaxed ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                            The **Air Quality Index (AQI)** is a unified score that tells you how clean or polluted the air is.
                            It is calculated based on the highest sub-index of key pollutants (PM2.5, PM10, NO2, etc.).
                            In India, the **CPCB (Central Pollution Control Board)** defines the standards used in this app.
                        </p>
                    </section>

                    <section>
                        <h3 className="text-lg font-semibold mb-3">🌈 AQI Categories (CPCB India)</h3>
                        <div className="grid gap-2 text-sm">
                            <div className="grid grid-cols-12 gap-2 items-center p-2 rounded bg-green-500/10 border-l-4 border-green-500">
                                <span className="col-span-3 font-bold text-green-600 dark:text-green-400">0 - 50</span>
                                <span className="col-span-3 font-bold">Good</span>
                                <span className="col-span-6 opacity-80">Minimal impact.</span>
                            </div>
                            <div className="grid grid-cols-12 gap-2 items-center p-2 rounded bg-lime-500/10 border-l-4 border-lime-500">
                                <span className="col-span-3 font-bold text-lime-600 dark:text-lime-400">51 - 100</span>
                                <span className="col-span-3 font-bold">Satisfactory</span>
                                <span className="col-span-6 opacity-80">Minor breathing discomfort to sensitive people.</span>
                            </div>
                            <div className="grid grid-cols-12 gap-2 items-center p-2 rounded bg-yellow-500/10 border-l-4 border-yellow-500">
                                <span className="col-span-3 font-bold text-yellow-600 dark:text-yellow-400">101 - 200</span>
                                <span className="col-span-3 font-bold">Moderate</span>
                                <span className="col-span-6 opacity-80">Breathing discomfort with lung/heart disease.</span>
                            </div>
                            <div className="grid grid-cols-12 gap-2 items-center p-2 rounded bg-orange-500/10 border-l-4 border-orange-500">
                                <span className="col-span-3 font-bold text-orange-600 dark:text-orange-400">201 - 300</span>
                                <span className="col-span-3 font-bold">Poor</span>
                                <span className="col-span-6 opacity-80">Breathing discomfort on prolonged exposure.</span>
                            </div>
                            <div className="grid grid-cols-12 gap-2 items-center p-2 rounded bg-red-500/10 border-l-4 border-red-500">
                                <span className="col-span-3 font-bold text-red-600 dark:text-red-400">301 - 400</span>
                                <span className="col-span-3 font-bold">Very Poor</span>
                                <span className="col-span-6 opacity-80">Respiratory illness on prolonged exposure.</span>
                            </div>
                            <div className="grid grid-cols-12 gap-2 items-center p-2 rounded bg-purple-600/10 border-l-4 border-purple-600">
                                <span className="col-span-3 font-bold text-purple-600 dark:text-purple-400">400+</span>
                                <span className="col-span-3 font-bold">Severe</span>
                                <span className="col-span-6 opacity-80">Affects healthy people; serious impact on existing diseases.</span>
                            </div>
                        </div>
                    </section>

                    <section className="space-y-4">
                        <h3 className="text-lg font-semibold">🏭 Common Pollutants</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                            <div className={`p-3 rounded-xl border ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-gray-100 bg-gray-50'}`}>
                                <h4 className="font-bold mb-1">PM2.5 (Fine Particulate Matter)</h4>
                                <p className="opacity-70">Tiny particles less than 2.5 microns. Penetrates deep into lungs and bloodstream. Sources: Vehicles, industry, burning.</p>
                            </div>
                            <div className={`p-3 rounded-xl border ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-gray-100 bg-gray-50'}`}>
                                <h4 className="font-bold mb-1">PM10 (Coarse Particulate Matter)</h4>
                                <p className="opacity-70">Larger particles like dust. Irritates eyes, nose, and throat. Sources: Construction dust, road dust, wind.</p>
                            </div>
                            <div className={`p-3 rounded-xl border ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-gray-100 bg-gray-50'}`}>
                                <h4 className="font-bold mb-1">NO2 (Nitrogen Dioxide)</h4>
                                <p className="opacity-70">Gas from burning fuel. Inflames airway. Sources: Car exhaust (traffic), power plants.</p>
                            </div>
                            <div className={`p-3 rounded-xl border ${isDarkMode ? 'border-slate-700 bg-slate-800' : 'border-gray-100 bg-gray-50'}`}>
                                <h4 className="font-bold mb-1">O3 (Ozone)</h4>
                                <p className="opacity-70">Ground-level ozone. Triggers asthma. Sources: Reaction of sunlight with pollutants.</p>
                            </div>
                        </div>
                    </section>
                </div>
            </div>
        </div>
    );
}
