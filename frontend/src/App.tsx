import { useEffect, useCallback, useRef } from 'react';
import { useStore } from './store';
import { fetchCurrentAQI, fetchPredictions, clearAPICache, fetchStationsInBounds } from './api';
import { generateMockPredictions } from './utils';
import type { Station } from './data/cities';

// Components
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import AQICard from './components/AQICard';
import PollutantCards from './components/PollutantCards';
import HealthSection from './components/HealthSection';
import PredictionChart from './components/PredictionChart';
import ActivitySection from './components/ActivitySection';
import HealthAlert from './components/HealthAlert';
import LoadingState from './components/LoadingState';
import StationList from './components/StationList';
import FloatingHelp from './components/FloatingHelp';

// Pages
import HeatmapPage from './components/HeatmapPage';
import AnalyticsPage from './components/AnalyticsPage';
import ReportsPage from './components/ReportsPage';
import SettingsPage from './components/SettingsPage';
import ComparePage from './components/ComparePage';

function App() {
    const {
        city,
        setCity,
        selectedStation,
        setSelectedStation,
        currentData,
        setCurrentData,
        setPredictions,
        scenario,
        isLoading,
        setIsLoading,
        error,
        setError,
        setLastUpdate,
        refreshCountdown,
        setRefreshCountdown,
        settings,

        activePage,
        allStations,
        setAllStations,
    } = useStore();

    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const isDarkMode = settings.isDarkMode;

    // Load data for current location
    const loadData = useCallback(
        async (cityName: string, station?: Station, forceRefresh: boolean = false) => {
            setIsLoading(true);
            setError(null);

            try {
                if (forceRefresh) {
                    clearAPICache();
                }

                console.log(`[App] Fetching AQI data for: ${cityName}`, station ? `Station: ${station.name}` : '');

                // Fetch current AQI with health conditions for personalization
                const aqiData = await fetchCurrentAQI(cityName, station, settings.healthConditions);

                if (aqiData.city && aqiData.city !== cityName) {
                    setCity(aqiData.city);
                }

                setCurrentData(aqiData);
                console.log(`[App] AQI loaded: ${aqiData.current.aqi_value} (${aqiData.current.aqi_source})`);

                // Fetch predictions
                const predictionData = await fetchPredictions(cityName, scenario, aqiData.current.aqi_value);
                setPredictions(predictionData);

                setLastUpdate(new Date());
                setRefreshCountdown(settings.refreshInterval);
            } catch (err) {
                console.error('[App] Error loading data:', err);
                setError(err instanceof Error ? err.message : 'Failed to load data');
            } finally {
                setIsLoading(false);
            }
        },
        [scenario, settings.refreshInterval, settings.healthConditions, setCurrentData, setError, setIsLoading, setLastUpdate, setCity, setPredictions, setRefreshCountdown]
    );

    // Update predictions when scenario changes
    useEffect(() => {
        if (currentData) {
            const newPredictions = generateMockPredictions(currentData.current.aqi_value, scenario);
            setPredictions(newPredictions);
        }
    }, [scenario, currentData, setPredictions]);

    // Re-fetch when health conditions change
    useEffect(() => {
        if (currentData && city) {
            loadData(city, selectedStation || undefined, false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [settings.healthConditions]);

    // Initial load
    useEffect(() => {
        loadData(city, selectedStation || undefined, true);

        // Background fetch of all stations for search/heatmap if not already loaded
        if (allStations.length === 0) {
            fetchStationsInBounds().then(stations => {
                if (stations && stations.length > 0) {
                    setAllStations(stations);
                }
            });
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Auto-refresh countdown
    useEffect(() => {
        intervalRef.current = setInterval(() => {
            setRefreshCountdown(Math.max(0, refreshCountdown - 1));

            if (refreshCountdown <= 1) {
                loadData(city, selectedStation || undefined, false);
            }
        }, 1000);

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [refreshCountdown, city, selectedStation, loadData, setRefreshCountdown]);

    // Handle city search/selection
    const handleCitySearch = useCallback(
        (newCity: string, station?: Station) => {
            console.log('[App] City changed to:', newCity, station?.name || '');
            setCity(newCity);
            setSelectedStation(station || null);
            loadData(newCity, station, true);
        },
        [loadData, setCity, setSelectedStation]
    );

    // Handle location detection
    const handleDetectLocation = useCallback(() => {
        if (!navigator.geolocation) {
            setError('Geolocation is not supported by your browser');
            return;
        }

        setIsLoading(true);
        navigator.geolocation.getCurrentPosition(
            async (position) => {
                const { findNearestCity } = await import('./data/cities');
                const nearest = findNearestCity(position.coords.latitude, position.coords.longitude);
                console.log('[App] Detected nearest city:', nearest.name);
                setCity(nearest.name);
                setSelectedStation(null);
                loadData(nearest.name, undefined, true);
            },
            (geoError) => {
                console.error('[App] Geolocation error:', geoError);
                setError('Unable to detect location. Using default city.');
                loadData(city, selectedStation || undefined, false);
            },
            { timeout: 5000, maximumAge: 600000 }
        );
    }, [loadData, city, selectedStation, setError, setIsLoading, setCity, setSelectedStation]);

    // Apply dark mode
    useEffect(() => {
        document.body.classList.toggle('dark', isDarkMode);
        document.documentElement.classList.toggle('dark', isDarkMode);
    }, [isDarkMode]);

    // Render page content
    const renderPageContent = () => {
        switch (activePage) {
            case 'heatmap':
                return <HeatmapPage />;
            case 'analytics':
                return <AnalyticsPage />;
            case 'reports':
                return <ReportsPage />;
            case 'settings':
                return <SettingsPage />;
            case 'compare':
                return <ComparePage />;
            default:
                return renderDashboard();
        }
    };

    // Render dashboard
    const renderDashboard = () => {
        if (isLoading && !currentData) {
            return <LoadingState />;
        }

        return (
            <div className="space-y-6 animate-fade-in">
                {/* Error */}
                {error && (
                    <div className={`p-4 rounded-xl ${isDarkMode ? 'bg-red-900/30 text-red-400' : 'bg-red-100 text-red-700'}`}>
                        <p className="font-medium">⚠️ {error}</p>
                        <button
                            onClick={() => loadData(city, selectedStation || undefined, true)}
                            className="mt-2 text-sm underline hover:no-underline"
                        >
                            Try again
                        </button>
                    </div>
                )}

                {/* Health Alert */}
                {currentData?.personal_risk && settings.enableNotifications && currentData.current.aqi_value > settings.alertThreshold && (
                    <HealthAlert risk={currentData.personal_risk} />
                )}

                {/* Main AQI + Pollutants */}
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 md:gap-6">
                    <AQICard />
                    <div className="lg:col-span-3">
                        <PollutantCards />
                    </div>
                </div>

                {/* Station List for City View */}
                {!selectedStation && (
                    <div className="animate-fade-in delay-200">
                        <StationList city={city} />
                    </div>
                )}

                {/* Health Recommendations */}
                <HealthSection />

                {/* 24-Hour Prediction */}
                <PredictionChart />

                {/* Activity Recommendations */}
                <ActivitySection />
            </div>
        );
    };

    return (
        <div className={`flex min-h-screen ${isDarkMode ? 'dark bg-slate-900' : 'bg-gray-50'}`}>
            <Sidebar />

            <main className="flex-1 p-4 md:p-6 lg:p-8 overflow-y-auto">
                {/* Header - only on dashboard */}
                {activePage === 'dashboard' && (
                    <Header onSearch={handleCitySearch} onDetectLocation={handleDetectLocation} />
                )}

                {/* Back button for non-dashboard pages (except heatmap which has its own header) */}
                {activePage !== 'dashboard' && activePage !== 'heatmap' && activePage !== 'compare' && (
                    <div className="mb-6">
                        <button
                            onClick={() => useStore.getState().setActivePage('dashboard')}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${isDarkMode ? 'bg-slate-800 text-gray-300 hover:bg-slate-700' : 'bg-white text-gray-700 hover:bg-gray-100'
                                } border ${isDarkMode ? 'border-slate-700' : 'border-gray-200'}`}
                        >
                            ← Back to Dashboard
                        </button>
                    </div>
                )}

                {renderPageContent()}
            </main>

            <FloatingHelp />
        </div>
    );
}

export default App;
