import { useMemo } from 'react';
import { Heart, User, Wind, Info, Cigarette } from 'lucide-react';
import { useStore } from '../store';

interface AqiInsightsStripProps {
    aqi: number;
    dominantPollutant: string;
    pm25Value: number | null;
    location: string;
    isDarkMode: boolean;
}

export default function AqiInsightsStrip({ aqi, dominantPollutant, pm25Value, location, isDarkMode }: AqiInsightsStripProps) {
    const { settings } = useStore();
    const userHealthConditions = settings.healthConditions;

    // 1. Personalized Health Note (Prioritized)
    const personalNote = useMemo(() => {
        const hasRespiratory = userHealthConditions.some(c => ['asthma', 'respiratory'].includes(c));
        const hasHeart = userHealthConditions.some(c => ['heart_disease'].includes(c));

        if (hasRespiratory && aqi >= 101) return "Asthma risk elevated today – keep inhaler ready & prefer indoor activities.";
        if (hasHeart && aqi >= 151) return "Cardiac risk higher – avoid strenuous activity outdoors.";
        return null;
    }, [userHealthConditions, aqi]);

    // 2. Dynamic Banner Content based on AQI Range (WAQI standard categories)
    const bannerInfo = useMemo(() => {
        if (aqi <= 50) return {
            bg: isDarkMode ? 'bg-green-900/30' : 'bg-green-50',
            text: isDarkMode ? 'text-green-300' : 'text-green-800',
            msg: "Excellent air quality today. Ideal for outdoor activities — minimal risk for everyone."
        };
        if (aqi <= 100) return {
            bg: isDarkMode ? 'bg-yellow-900/30' : 'bg-yellow-50',
            text: isDarkMode ? 'text-yellow-300' : 'text-yellow-800',
            msg: "Acceptable for most people. Sensitive groups (asthma, elderly, children) may notice minor effects."
        };
        if (aqi <= 150) return {
            bg: isDarkMode ? 'bg-orange-900/30' : 'bg-orange-50',
            text: isDarkMode ? 'text-orange-300' : 'text-orange-800',
            msg: "Sensitive people may experience breathing issues. Limit prolonged outdoor exertion."
        };
        if (aqi <= 200) return {
            bg: isDarkMode ? 'bg-red-900/30' : 'bg-red-50',
            text: isDarkMode ? 'text-red-300' : 'text-red-800',
            msg: "Everyone may begin to feel effects; sensitive groups more seriously. Reduce outdoor time."
        };
        if (aqi <= 300) return {
            bg: isDarkMode ? 'bg-purple-900/30' : 'bg-purple-50',
            text: isDarkMode ? 'text-purple-300' : 'text-purple-800',
            msg: "Health alert: increased risk of respiratory & heart issues. Stay indoors when possible."
        };
        return {
            bg: isDarkMode ? 'bg-rose-900/30' : 'bg-rose-50',
            text: isDarkMode ? 'text-rose-300' : 'text-rose-800',
            msg: "Emergency conditions — serious health effects likely. Avoid going outdoors."
        };
    }, [aqi, isDarkMode]);

    // 3. Cigarette Calculation (Berkeley Earth)
    const cigarettes = useMemo(() => {
        if (dominantPollutant === 'PM2.5' && aqi >= 50 && pm25Value !== null) {
            const cigs = Math.round(pm25Value / 22);
            if (cigs < 1) return null;
            return cigs > 4 ? '4+' : `${cigs}`; // Requirement: Show "~" for 1-4, but example said "≈ X". The user said: Show "~" for 1–4, "4+" for higher. And "≈ X cigarettes/day".
            // Let's refine based on "≈ X cigarettes/day equivalent"
        }
        return null;
    }, [aqi, dominantPollutant, pm25Value]);

    const cigaretteText = useMemo(() => {
        if (!cigarettes) return null;
        const countDisplay = cigarettes === '4+' ? '4+' : `~${cigarettes}`;
        return `≈ ${countDisplay} cigarettes/day equivalent (Berkeley Earth research)`;
    }, [cigarettes]);


    // 4. Smart Source Text
    const smartSource = useMemo(() => {
        const month = new Date().getMonth() + 1;
        const loc = location.toLowerCase();
        let text = "";

        if (dominantPollutant === 'PM2.5' && aqi > 100) {
            if (loc.includes('mumbai') || loc.includes('delhi') || loc.includes('bangalore') || loc.includes('chennai')) {
                text = "Likely: vehicle emissions + biomass/coal burning. Fine particles reach deep into lungs.";
            } else {
                text = "Likely: traffic, industry or burning. Can affect breathing.";
            }
        } else if (dominantPollutant === 'PM10') {
            text = "Likely: construction dust, road dust or windblown soil. Irritates eyes & airways.";
        } else if (dominantPollutant === 'O3' && month >= 3 && month <= 6) {
            text = "Likely: summer photochemical smog from traffic & industry.";
        } else if ((month >= 10 || month <= 2) && (loc.includes('karnataka') || loc.includes('punjab') || loc.includes('delhi') || loc.includes('haryana'))) {
            text = "Likely: seasonal crop burning contribution + winter inversion.";
        } else if (dominantPollutant === 'NO2') {
            text = "Likely: High traffic congestion or industrial exhaust.";
        } else {
            text = "Common urban/rural sources.";
        }
        return text;
    }, [dominantPollutant, aqi, location]);

    // 5. Visual Health Risk Icons Logic

    // Per requirements:
    // Good -> all gray opacity 50% (handled by else of threshold check if range is low)
    // Moderate -> sensitive icon yellow-500
    // Unhealthy for Sensitive+ -> lungs & heart orange/red-500, sensitive red-600

    // Refined logic for precise requirements:
    const getSensitiveIconClass = () => {
        if (aqi <= 50) return 'opacity-50 grayscale text-gray-400';
        if (aqi <= 100) return 'opacity-100 text-yellow-500'; // Moderate
        return 'opacity-100 text-red-600'; // Unhealthy Sensitive+
    };

    const getOrganIconClass = (color: string) => {
        if (aqi <= 100) return 'opacity-50 grayscale text-gray-400'; // Good/Moderate -> Gray
        return `opacity-100 ${color}`; // Unhealthy Sensitive+ -> colored
    };

    return (
        <div className="space-y-4">
            {/* Health Icons Row */}
            <div className="flex justify-center gap-4 mb-2">
                <div className={`flex flex-col items-center group cursor-help transition-all ${getOrganIconClass('text-orange-500')}`}>
                    <Wind className="w-6 h-6" />
                    <span className="text-[10px] mt-1 opacity-60">Lungs</span>
                </div>
                <div className={`flex flex-col items-center group cursor-help transition-all ${getOrganIconClass('text-red-500')}`}>
                    <Heart className="w-6 h-6" />
                    <span className="text-[10px] mt-1 opacity-60">Heart</span>
                </div>
                <div className={`flex flex-col items-center group cursor-help transition-all ${getSensitiveIconClass()}`}>
                    <User className="w-6 h-6" />
                    <span className="text-[10px] mt-1 opacity-60">Sensitive</span>
                </div>
            </div>

            {/* Banner Block */}
            <div className={`px-4 py-3 rounded-xl ${bannerInfo.bg} ${bannerInfo.text} text-base transition-colors relative`}>
                {/* Priority Personal Note */}
                {personalNote && (
                    <div className="mb-2 pb-2 border-b border-current border-opacity-20 font-bold flex items-start gap-2">
                        <Info className="w-5 h-5 shrink-0 mt-0.5" />
                        {personalNote}
                    </div>
                )}

                <div className="font-medium">
                    {bannerInfo.msg}
                </div>

                {/* Cigarette Equivalence Line */}
                {cigaretteText && (
                    <div className="mt-2 text-sm text-gray-600 dark:text-gray-400 italic flex items-center gap-1.5 border-t border-black/5 dark:border-white/5 pt-1.5">
                        <Cigarette className="w-4 h-4 opacity-70" />
                        {cigaretteText}
                    </div>
                )}
            </div>

            {/* Smart Source Line */}
            <div className={`px-4 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                <p>
                    Dominant: <span className="font-semibold text-brand-primary">{dominantPollutant}</span>
                    <span className="mx-2">•</span>
                    {smartSource}
                </p>
            </div>
        </div>
    );
}
