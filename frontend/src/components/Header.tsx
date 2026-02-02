import { useState, useEffect, useRef, useCallback } from 'react';
import { Search, MapPin, RefreshCw, Clock, ChevronDown, X, Navigation } from 'lucide-react';
import { useStore } from '../store';
import { searchLocations, getCityByName, type Station } from '../data/cities';

import { formatTime, formatCountdown, debounce } from '../utils';

interface HeaderProps {
    onSearch: (city: string, station?: Station) => void;
    onDetectLocation: () => void;
}

export default function Header({ onSearch, onDetectLocation }: HeaderProps) {
    const { city, selectedStation, currentData, lastUpdate, refreshCountdown, settings, allStations } = useStore();
    const isDarkMode = settings.isDarkMode;

    const [searchValue, setSearchValue] = useState('');
    const [suggestions, setSuggestions] = useState<Array<{ type: 'city' | 'station'; name: string; city?: string; state: string; id?: string; stationData?: any }>>([]);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [showStationDropdown, setShowStationDropdown] = useState(false);
    const [highlightedIndex, setHighlightedIndex] = useState(-1);

    const searchRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // Get available stations for current city
    const cityData = getCityByName(city);
    const stations = cityData?.stations || [];
    const hasMultipleStations = stations.length > 1;

    // Debounced search
    const debouncedSearch = useCallback(
        debounce(async (query: string) => {
            if (query.length >= 2) {
                const cityResults = searchLocations(query);

                // Search in allStations (WAQI bounds data)
                const stationResults = allStations
                    .filter((s: any) => s.station?.name?.toLowerCase().includes(query.toLowerCase()))
                    .map((s: any) => ({
                        type: 'station' as const,
                        name: s.station.name,
                        city: s.station.name.split(',')[0], // Approximation
                        state: 'India',
                        id: String(s.uid),
                        stationData: s
                    }))
                    .slice(0, 5);

                // Merge: City results first, then unique stations (Local Only)
                const seen = new Set(cityResults.map(r => r.name.toLowerCase()));

                const uniqueStations = stationResults.filter(s => {
                    if (seen.has(s.name.toLowerCase())) return false;
                    seen.add(s.name.toLowerCase());
                    return true;
                });

                setSuggestions([...cityResults, ...uniqueStations].slice(0, 15));
                setShowSuggestions(true);
                setHighlightedIndex(-1);
            } else {
                setSuggestions([]);
                setShowSuggestions(false);
            }
        }, 300),
        [allStations]
    );

    // Handle input change
    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value;
        setSearchValue(value);
        debouncedSearch(value);
    };

    // Handle suggestion select
    const handleSuggestionSelect = (suggestion: (typeof suggestions)[0]) => {
        if (suggestion.type === 'station') {
            if (suggestion.stationData) {
                // If derived from WAQI allStations, construct a Station object
                const s = suggestion.stationData;
                const stationObj: Station = {
                    id: String(s.uid),
                    name: s.station.name,
                    city: s.station.name.split(',')[0], // Approximation
                    state: 'India', // Best effort
                    lat: s.lat,
                    lon: s.lon
                };
                onSearch(suggestion.city || city, stationObj);
            } else {
                // Fallback for existing station logic from cities.ts
                const stationCity = getCityByName(suggestion.city || '');
                const station = stationCity?.stations.find((s) => s.id === suggestion.id);
                onSearch(suggestion.city || city, station);
            }
        } else {
            onSearch(suggestion.name);
        }
        setSearchValue('');
        setShowSuggestions(false);
        inputRef.current?.blur();
    };

    // Handle station select
    const handleStationSelect = (station: Station | null) => {
        onSearch(city, station || undefined);
        setShowStationDropdown(false);
    };

    // Keyboard navigation
    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (!showSuggestions) return;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setHighlightedIndex((prev) => Math.min(prev + 1, suggestions.length - 1));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setHighlightedIndex((prev) => Math.max(prev - 1, -1));
        } else if (e.key === 'Enter' && highlightedIndex >= 0) {
            e.preventDefault();
            handleSuggestionSelect(suggestions[highlightedIndex]);
        } else if (e.key === 'Escape') {
            setShowSuggestions(false);
        }
    };

    // Close dropdowns on outside click
    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
                setShowSuggestions(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <header className="mb-6">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                {/* Search Section */}
                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 flex-1">
                    {/* Search Bar */}
                    <div className="relative flex-1 max-w-md" ref={searchRef}>
                        <Search className={`absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 ${isDarkMode ? 'text-gray-400' : 'text-gray-400'}`} />
                        <input
                            ref={inputRef}
                            type="text"
                            value={searchValue}
                            onChange={handleInputChange}
                            onKeyDown={handleKeyDown}
                            onFocus={() => searchValue.length >= 2 && setShowSuggestions(true)}
                            placeholder="Search city or station..."
                            className={`w-full pl-10 pr-10 py-3 rounded-xl border-2 transition-all ${isDarkMode
                                ? 'bg-slate-800 border-slate-700 text-white placeholder-gray-500 focus:border-brand-primary'
                                : 'bg-white border-gray-200 text-gray-900 placeholder-gray-400 focus:border-brand-primary'
                                } focus:outline-none`}
                        />
                        {searchValue && (
                            <button
                                onClick={() => {
                                    setSearchValue('');
                                    setSuggestions([]);
                                    setShowSuggestions(false);
                                }}
                                className={`absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded-full ${isDarkMode ? 'hover:bg-slate-700' : 'hover:bg-gray-100'
                                    }`}
                            >
                                <X className="w-4 h-4" />
                            </button>
                        )}

                        {/* Suggestions Dropdown */}
                        {showSuggestions && suggestions.length > 0 && (
                            <div className={`absolute top-full left-0 right-0 mt-2 rounded-xl shadow-xl border overflow-hidden z-50 ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-gray-200'
                                }`}>
                                {suggestions.map((suggestion, index) => (
                                    <button
                                        key={`${suggestion.type}-${suggestion.name}-${index}`}
                                        onClick={() => handleSuggestionSelect(suggestion)}
                                        className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors ${index === highlightedIndex
                                            ? 'bg-brand-primary text-brand-dark'
                                            : isDarkMode
                                                ? 'hover:bg-slate-700 text-white'
                                                : 'hover:bg-gray-50 text-gray-900'
                                            }`}
                                    >
                                        <MapPin className="w-4 h-4 flex-shrink-0" />
                                        <div className="flex-1 min-w-0">
                                            <p className="font-medium truncate">{suggestion.name}</p>
                                            <p className={`text-xs truncate ${index === highlightedIndex ? 'text-brand-dark/70' : isDarkMode ? 'text-gray-400' : 'text-gray-500'
                                                }`}>
                                                {suggestion.type === 'station' ? `Station in ${suggestion.city}, ${suggestion.state}` : suggestion.state}
                                            </p>
                                        </div>
                                        <span className={`text-xs px-2 py-0.5 rounded-full ${suggestion.type === 'station'
                                            ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300'
                                            : 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300'
                                            }`}>
                                            {suggestion.type === 'station' ? 'Station' : 'City'}
                                        </span>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Location Button */}
                    <button
                        onClick={onDetectLocation}
                        className={`flex items-center gap-2 px-4 py-3 rounded-xl transition-colors ${isDarkMode
                            ? 'bg-slate-800 text-white hover:bg-slate-700 border border-slate-700'
                            : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-200'
                            }`}
                    >
                        <Navigation className="w-5 h-5" />
                        <span className="hidden sm:inline">Detect Location</span>
                    </button>
                </div>

                {/* Location & Status Info */}
                <div className="flex flex-wrap items-center gap-3">
                    {/* Current Location Badge */}
                    <div className={`flex items-center gap-2 px-4 py-2 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'
                        } border ${isDarkMode ? 'border-slate-700' : 'border-gray-200'}`}>
                        <MapPin className="w-4 h-4 text-brand-primary" />
                        <span className={`font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{city}</span>

                        {/* Station Selector (if multiple stations) */}
                        {hasMultipleStations && (
                            <div className="relative">
                                <button
                                    onClick={() => setShowStationDropdown(!showStationDropdown)}
                                    className={`flex items-center gap-1 ml-2 pl-2 border-l ${isDarkMode ? 'border-slate-600 text-gray-400' : 'border-gray-200 text-gray-500'
                                        } hover:text-brand-primary transition-colors`}
                                >
                                    <span className="text-sm">{selectedStation?.name || 'All Stations'}</span>
                                    <ChevronDown className="w-4 h-4" />
                                </button>

                                {showStationDropdown && (
                                    <div className={`absolute top-full right-0 mt-2 w-56 rounded-xl shadow-xl border overflow-hidden z-50 ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-gray-200'
                                        }`}>
                                        <button
                                            onClick={() => handleStationSelect(null)}
                                            className={`w-full px-4 py-2 text-left text-sm ${!selectedStation
                                                ? 'bg-brand-primary text-brand-dark font-medium'
                                                : isDarkMode
                                                    ? 'hover:bg-slate-700 text-white'
                                                    : 'hover:bg-gray-50 text-gray-900'
                                                }`}
                                        >
                                            All Stations
                                        </button>
                                        {stations.map((station) => (
                                            <button
                                                key={station.id}
                                                onClick={() => handleStationSelect(station)}
                                                className={`w-full px-4 py-2 text-left text-sm ${selectedStation?.id === station.id
                                                    ? 'bg-brand-primary text-brand-dark font-medium'
                                                    : isDarkMode
                                                        ? 'hover:bg-slate-700 text-white'
                                                        : 'hover:bg-gray-50 text-gray-900'
                                                    }`}
                                            >
                                                {station.name}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Live Update Time */}
                    {lastUpdate && (
                        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${isDarkMode ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-600'
                            }`}>
                            <Clock className="w-4 h-4" />
                            <span className="text-sm">Updated: {formatTime(new Date(lastUpdate))}</span>
                        </div>
                    )}

                    {/* Refresh Countdown */}
                    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${isDarkMode ? 'bg-slate-800' : 'bg-gray-100'
                        }`}>
                        <RefreshCw className={`w-4 h-4 ${refreshCountdown <= 10 ? 'animate-spin text-brand-primary' : isDarkMode ? 'text-gray-400' : 'text-gray-500'}`} />
                        <span className={`text-sm font-mono ${refreshCountdown <= 10 ? 'text-brand-primary font-bold' : isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                            {formatCountdown(refreshCountdown)}
                        </span>
                    </div>

                    {/* Data Source */}
                    {currentData?.current.aqi_source && (
                        <div className={`hidden lg:flex items-center gap-1 text-xs px-2 py-1 rounded-full ${isDarkMode ? 'bg-green-900/30 text-green-400' : 'bg-green-100 text-green-700'
                            }`}>
                            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                            {currentData.current.aqi_source}
                        </div>
                    )}
                </div>
            </div>
        </header>
    );
}
