import { LayoutDashboard, Map, BarChart3, FileText, Settings, Moon, Sun, Leaf } from 'lucide-react';
import { useStore } from '../store';
import type { PageType } from '../types';

interface NavItem {
    id: PageType;
    label: string;
    icon: React.ReactNode;
}

const navItems: NavItem[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-5 h-5" /> },
    { id: 'heatmap', label: 'India Heatmap', icon: <Map className="w-5 h-5" /> },
    { id: 'analytics', label: 'Analytics', icon: <BarChart3 className="w-5 h-5" /> },
    { id: 'reports', label: 'Reports', icon: <FileText className="w-5 h-5" /> },
    { id: 'settings', label: 'Settings', icon: <Settings className="w-5 h-5" /> },
];

export default function Sidebar() {
    const { activePage, setActivePage, settings, updateSettings } = useStore();
    const isDarkMode = settings.isDarkMode;

    const toggleDarkMode = () => {
        updateSettings({ isDarkMode: !isDarkMode });
    };

    return (
        <aside
            className={`w-64 min-h-screen flex flex-col ${isDarkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-gray-200'
                } border-r transition-colors duration-300`}
        >
            {/* Logo */}
            <div className="p-6 border-b border-inherit">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-primary to-green-500 flex items-center justify-center">
                        <span className="text-2xl">🍃</span>
                    </div>
                    <div>
                        <h1 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                            🍃 AeroClean
                        </h1>
                        <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                            Air Quality Monitor
                        </p>
                    </div>
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 py-6 px-3 space-y-1">
                {navItems.map((item) => {
                    const isActive = activePage === item.id;
                    return (
                        <button
                            key={item.id}
                            onClick={() => setActivePage(item.id)}
                            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all duration-200 ${isActive
                                ? 'bg-brand-primary text-brand-dark shadow-lg shadow-brand-primary/20'
                                : isDarkMode
                                    ? 'text-gray-400 hover:text-white hover:bg-slate-800'
                                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                                }`}
                        >
                            {item.icon}
                            {item.label}
                            {item.id === 'heatmap' && (
                                <span className="ml-auto px-2 py-0.5 text-xs bg-green-500/20 text-green-500 rounded-full">
                                    Live
                                </span>
                            )}
                        </button>
                    );
                })}
            </nav>

            {/* Dark Mode Toggle */}
            <div className={`p-4 border-t ${isDarkMode ? 'border-slate-800' : 'border-gray-200'}`}>
                <button
                    onClick={toggleDarkMode}
                    className={`w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all ${isDarkMode ? 'bg-slate-800 text-white' : 'bg-gray-100 text-gray-700'
                        }`}
                >
                    <div className="flex items-center gap-3">
                        {isDarkMode ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
                        <span className="font-medium">{isDarkMode ? 'Dark Mode' : 'Light Mode'}</span>
                    </div>
                    <div
                        className={`w-10 h-6 rounded-full relative ${isDarkMode ? 'bg-brand-primary' : 'bg-gray-300'}`}
                    >
                        <span
                            className={`absolute top-1 w-4 h-4 rounded-full bg-white shadow transition-all ${isDarkMode ? 'left-5' : 'left-1'
                                }`}
                        />
                    </div>
                </button>
            </div>

            {/* Footer */}
            <div className={`p-4 ${isDarkMode ? 'text-gray-600' : 'text-gray-400'}`}>
                <div className="flex items-center gap-2 text-xs">
                    <Leaf className="w-4 h-4" />
                    <span>AeroClean v2.1</span>
                </div>
            </div>
        </aside>
    );
}
