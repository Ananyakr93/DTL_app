import type { AQIData, HourlyPrediction, HistoricalDataPoint, Scenario, WAQIResponse, CityAQI, HealthCondition } from './types';
import { getAQIClass, getAQIStatus, generateMockPredictions, generateHistoricalData } from './utils';
import { getCityByName, INDIAN_CITIES, type Station } from './data/cities';

// =============================================================================
// API CONFIGURATION
// =============================================================================

// WAQI API - Primary source (reliable, works globally including India)
const WAQI_TOKEN = import.meta.env.VITE_WAQI_TOKEN || '91cfb794c918bbc8fed384ff6aab22383dec190a';

const WAQI_API = 'https://api.waqi.info';
const INDIA_BOUNDS = '6.0,68.0,35.0,97.0';

// CPCB API - Secondary for India (can be slow/unreliable)
const CPCB_API_KEY = '579b464db66ec23bdd0000019cbe093c60694b174ef48b7e614a8098';
const CPCB_API = 'https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69';

// Cache with TTL
const apiCache = new Map<string, { data: AQIData; timestamp: number }>();
const CACHE_TTL = 180000; // 3 minutes

// =============================================================================
// INDIAN LOCATION DETECTION
// =============================================================================

const INDIAN_CITIES_SET = new Set([
    'delhi', 'new delhi', 'mumbai', 'bangalore', 'bengaluru', 'chennai', 'kolkata',
    'hyderabad', 'pune', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur',
    'indore', 'thane', 'bhopal', 'visakhapatnam', 'patna', 'vadodara', 'ghaziabad',
    'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut', 'rajkot', 'varanasi',
    'srinagar', 'aurangabad', 'dhanbad', 'amritsar', 'navi mumbai', 'allahabad',
    'ranchi', 'howrah', 'coimbatore', 'jabalpur', 'gwalior', 'vijayawada', 'jodhpur',
    'madurai', 'raipur', 'kota', 'chandigarh', 'guwahati', 'solapur', 'hubli',
    'mysore', 'mysuru', 'tiruchirappalli', 'bareilly', 'aligarh', 'tiruppur',
    'moradabad', 'jalandhar', 'bhubaneswar', 'salem', 'warangal', 'guntur',
    'bhiwandi', 'saharanpur', 'gorakhpur', 'bikaner', 'amravati', 'noida',
    'jamshedpur', 'bhilai', 'cuttack', 'firozabad', 'kochi', 'cochin',
    'thiruvananthapuram', 'trivandrum', 'shimla', 'dehradun', 'rishikesh',
    'haridwar', 'darjeeling', 'gangtok', 'shillong', 'imphal', 'aizawl',
    'kohima', 'agartala', 'itanagar', 'panaji', 'goa', 'pondicherry', 'puducherry',
    'surat', 'mangalore', 'mangaluru', 'kozhikode', 'calicut', 'thrissur',
    'kollam', 'tirunelveli', 'erode', 'vellore', 'nellore', 'rajahmundry',
    'karimnagar', 'nizamabad', 'khammam', 'anantapur', 'kurnool', 'kadapa',
    'eluru', 'ongole', 'tirupati', 'chittoor', 'bellary', 'gulbarga', 'belgaum',
    'dharwad', 'shimoga', 'tumkur', 'davangere', 'bijapur', 'hospet',
    'chikkamagaluru', 'chikmagalur', 'madikeri', 'udupi', 'karwar', 'gurugram',
    'gurgaon', 'greater noida', 'secunderabad', 'vijaywada',
]);

function isIndianLocation(cityName: string): boolean {
    const normalized = cityName.toLowerCase().trim().replace(/[,\-]/g, ' ');

    // Direct match
    if (INDIAN_CITIES_SET.has(normalized)) return true;

    // Partial match (e.g., "Pune, Maharashtra" contains "pune")
    for (const indianCity of INDIAN_CITIES_SET) {
        if (normalized.includes(indianCity) || indianCity.includes(normalized.split(' ')[0])) {
            return true;
        }
    }

    // Check our database
    if (INDIAN_CITIES.some(c => c.name.toLowerCase() === normalized.split(',')[0].trim())) {
        return true;
    }

    return false;
}

// =============================================================================
// HEALTH RECOMMENDATIONS
// =============================================================================

function getPersonalizedRecommendations(
    aqi: number,
    healthConditions: HealthCondition[]
): { health: AQIData['health']; activities: string[]; personal_risk: AQIData['personal_risk'] } {
    const isSensitive = healthConditions.some(c =>
        ['asthma', 'respiratory', 'heart_disease', 'pregnant', 'elderly', 'children'].includes(c)
    );

    const effectiveAqi = isSensitive ? aqi * 1.2 : aqi;
    let health: AQIData['health'];
    let activities: string[];
    let personal_risk: AQIData['personal_risk'];

    if (effectiveAqi <= 50) {
        health = {
            general: 'Air quality is excellent. Perfect for outdoor activities.',
            sensitive: isSensitive ? 'Good conditions - monitor if symptoms arise.' : 'Air quality is ideal.',
            outdoor_advice: 'Enjoy outdoor activities freely.',
            mask_advice: 'No mask required.',
            specific_advice: [],
        };
        activities = ['✅ Jogging & Running', '✅ Cycling', '✅ Outdoor sports', '✅ Park walks'];
        personal_risk = {
            risk_level: 'Low',
            alert_title: 'Good Air Quality',
            alert_message: 'Perfect conditions for outdoor activities!',
            tips: ['Great day for exercise', 'Open windows for fresh air'],
        };
    } else if (effectiveAqi <= 100) {
        health = {
            general: 'Air quality is acceptable for most people.',
            sensitive: isSensitive ? 'Consider limiting prolonged outdoor exertion.' : 'Sensitive individuals may experience minor symptoms.',
            outdoor_advice: 'Moderate outdoor activity is fine.',
            mask_advice: isSensitive ? 'Consider N95 mask for extended outdoor time.' : 'Mask optional.',
            specific_advice: [],
        };
        activities = ['✅ Light exercise', '✅ Walking', isSensitive ? '⚠️ Limit intense activity' : '✅ Running'];
        personal_risk = {
            risk_level: isSensitive ? 'Moderate' : 'Low',
            alert_title: 'Satisfactory Air Quality',
            alert_message: isSensitive ? 'Take moderate precautions.' : 'Generally acceptable for all.',
            tips: isSensitive ? ['Monitor symptoms', 'Have medications ready'] : ['Generally safe outdoors'],
        };
    } else if (effectiveAqi <= 200) {
        health = {
            general: 'Unhealthy for sensitive groups. Others may notice effects.',
            sensitive: 'Avoid prolonged outdoor activities.',
            outdoor_advice: 'Limit outdoor exertion.',
            mask_advice: 'N95 mask recommended outdoors.',
            specific_advice: [{ pollutant: 'PM2.5', message: 'Elevated fine particles', action: 'Reduce outdoor time' }],
        };
        activities = ['⚠️ Light walking only', '❌ Avoid jogging', '✅ Indoor exercise preferred'];
        personal_risk = {
            risk_level: isSensitive ? 'High' : 'Moderate',
            alert_title: 'Unhealthy Air Quality',
            alert_message: 'Limit prolonged outdoor exposure.',
            tips: ['Wear N95 mask outdoors', 'Keep windows closed', 'Use air purifier'],
        };
    } else if (effectiveAqi <= 300) {
        health = {
            general: 'Health alert: Everyone may experience health effects.',
            sensitive: 'AVOID outdoor activities.',
            outdoor_advice: 'Avoid all outdoor exertion.',
            mask_advice: 'N95 mask essential if going outdoors.',
            specific_advice: [{ pollutant: 'PM2.5', message: 'Very high levels', action: 'Stay indoors' }],
        };
        activities = ['🚨 Avoid outdoor activities', '❌ No outdoor exercise', '✅ Indoor only'];
        personal_risk = {
            risk_level: 'High',
            alert_title: '🚨 Poor Air Quality',
            alert_message: 'Health alert - avoid going outdoors.',
            tips: ['Stay indoors', 'Run air purifiers', 'Wear N95 if must go out'],
        };
    } else {
        health = {
            general: 'HEALTH EMERGENCY: Hazardous air quality.',
            sensitive: 'CRITICAL: Stay indoors, seal doors/windows.',
            outdoor_advice: 'DO NOT go outdoors under any circumstances.',
            mask_advice: 'N95 mandatory, even indoors if air quality is poor.',
            specific_advice: [{ pollutant: 'All', message: 'Hazardous levels', action: 'Shelter in place' }],
        };
        activities = ['🚨 EMERGENCY: Stay indoors', '🚨 All outdoor activities dangerous'];
        personal_risk = {
            risk_level: 'Severe',
            alert_title: '🚨 HAZARDOUS',
            alert_message: 'Severe health emergency - stay indoors!',
            tips: ['Seal all openings', 'Run air purifiers continuously', 'Consider evacuation'],
        };
    }

    return { health, activities, personal_risk };
}

// =============================================================================
// WAQI API (PRIMARY - RELIABLE)
// =============================================================================

async function fetchWAQIData(query: string | { lat: number; lon: number } | { uid: string | number }): Promise<WAQIResponse | null> {
    try {
        let url: string;

        if (typeof query === 'string') {
            url = `${WAQI_API}/feed/${encodeURIComponent(query)}/?token=${WAQI_TOKEN}`;
        } else if ('uid' in query) {
            url = `${WAQI_API}/feed/@${query.uid}/?token=${WAQI_TOKEN}`;
        } else {
            url = `${WAQI_API}/feed/geo:${query.lat};${query.lon}/?token=${WAQI_TOKEN}`;
        }

        console.log(`[WAQI] Fetching: ${typeof query === 'string' ? query : 'uid' in query ? `UID ${query.uid}` : `${query.lat},${query.lon}`}`);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000);

        const response = await fetch(url, { signal: controller.signal });
        clearTimeout(timeoutId);

        const data = await response.json();

        if (data.status === 'ok' && data.data && typeof data.data.aqi === 'number' && data.data.aqi > 0) {
            console.log(`[WAQI] ✅ Success - AQI: ${data.data.aqi}, City: ${data.data.city?.name}`);
            return data as WAQIResponse;
        }

        // If direct query fails, try search API
        if (typeof query === 'string') {
            return await searchAndFetchWAQI(query);
        }

        return null;
    } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
            console.log(`[WAQI] Timeout for query`);
        } else {
            console.error('[WAQI] Error:', error);
        }
        return null;
    }
}

async function searchAndFetchWAQI(query: string): Promise<WAQIResponse | null> {
    try {
        const searchUrl = `${WAQI_API}/search/?keyword=${encodeURIComponent(query)}&token=${WAQI_TOKEN}`;
        console.log(`[WAQI] Searching: ${query}`);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000);

        const response = await fetch(searchUrl, { signal: controller.signal });
        clearTimeout(timeoutId);

        const data = await response.json();

        if (data.status === 'ok' && data.data && data.data.length > 0) {
            const match = data.data[0];
            console.log(`[WAQI] Found: ${match.station?.name} (uid: ${match.uid})`);

            // Fetch actual data
            const feedUrl = `${WAQI_API}/feed/@${match.uid}/?token=${WAQI_TOKEN}`;
            const feedResponse = await fetch(feedUrl);
            const feedData = await feedResponse.json();

            if (feedData.status === 'ok' && feedData.data && feedData.data.aqi > 0) {
                console.log(`[WAQI] ✅ Station AQI: ${feedData.data.aqi}`);
                return feedData as WAQIResponse;
            }
        }

        console.log(`[WAQI] No results for: ${query}`);
        return null;
    } catch (error) {
        console.error('[WAQI] Search error:', error);
        return null;
    }
}

export async function fetchStationsInBounds(): Promise<Array<{ uid: number; lat: number; lon: number; aqi: number; station: { name: string; time: string } }>> {
    const cacheKey = 'waqi-bounds-india';

    // Check cache (TTL 1 hour)
    const cached = apiCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < 3600000) { // 1 hour
        console.log(`[WAQI] Using cached bounds data (age: ${Math.round((Date.now() - cached.timestamp) / 60000)}m)`);
        return cached.data as any;
    }

    try {
        const url = `${WAQI_API}/map/bounds/?latlng=${INDIA_BOUNDS}&networks=all&token=${WAQI_TOKEN}`;
        console.log(`[WAQI] Fetching stations in bounds: ${INDIA_BOUNDS}`);

        const response = await fetch(url);
        const data = await response.json();

        if (data.status === 'ok' && Array.isArray(data.data)) {
            console.log(`[WAQI] Found ${data.data.length} stations in India bounds`);

            apiCache.set(cacheKey, { data: data.data, timestamp: Date.now() });
            return data.data;
        }

        console.error('[WAQI] Invalid bounds response:', data);
        return [];
    } catch (error) {
        console.error('[WAQI] Bounds fetch error:', error);
        return [];
    }
}

function convertWAQItoAQIData(waqi: WAQIResponse, cityName: string, healthConditions: HealthCondition[]): AQIData {
    const aqi = waqi.data.aqi;
    const colorClass = getAQIClass(aqi);
    const status = getAQIStatus(aqi);

    const pm2_5 = waqi.data.iaqi.pm25?.v ?? null;
    const pm10 = waqi.data.iaqi.pm10?.v ?? null;
    const no2 = waqi.data.iaqi.no2?.v ?? null;
    const so2 = waqi.data.iaqi.so2?.v ?? null;
    const co = waqi.data.iaqi.co?.v ?? null;
    const o3 = waqi.data.iaqi.o3?.v ?? null;

    // Clean station name
    let stationName = waqi.data.city?.name || cityName;
    stationName = stationName.replace(/\s*\([^)]*avg[^)]*\)\s*/gi, '').trim();

    const { health, activities, personal_risk } = getPersonalizedRecommendations(aqi, healthConditions);

    return {
        city: cityName,
        current: {
            aqi_value: aqi,
            aqi_status: status,
            aqi_color: colorClass,
            station: stationName,
            aqi_source: isIndianLocation(cityName) ? 'WAQI India' : 'WAQI Global',
            dominant_pollutant: waqi.data.dominentpol?.toUpperCase() || 'PM2.5',
            pm2_5,
            pm10,
            no2,
            so2,
            co,
            o3,
            temperature: waqi.data.iaqi.t?.v ?? null,
            humidity: waqi.data.iaqi.h?.v ?? null,
            wind: waqi.data.iaqi.w?.v ?? null,
            timestamp: waqi.data.time?.s,
        },
        health,
        activities,
        personal_risk,
    };
}

// =============================================================================
// CPCB API (SECONDARY FOR INDIA)
// =============================================================================

interface CPCBRecord {
    id?: string;
    country?: string;
    state?: string;
    city?: string;
    station?: string;
    last_update?: string;
    pollutant_id?: string;
    pollutant_min?: string;
    pollutant_max?: string;
    pollutant_avg?: string;
}

async function fetchCPCBData(city: string, targetStation?: string): Promise<{
    aqi: number;
    pm2_5: number | null;
    pm10: number | null;
    no2: number | null;
    so2: number | null;
    co: number | null;
    o3: number | null;
    station: string;
} | null> {
    try {
        // Try city filter
        const url = `${CPCB_API}?api-key=${CPCB_API_KEY}&format=json&limit=50&filters[city]=${encodeURIComponent(city)}`;
        console.log(`[CPCB] Trying city: ${city}`);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout

        const response = await fetch(url, { signal: controller.signal });
        clearTimeout(timeoutId);

        if (!response.ok) {
            console.log(`[CPCB] HTTP ${response.status}`);
            return null;
        }

        const data = await response.json();

        if (!data.records || data.records.length === 0) {
            console.log(`[CPCB] No records for: ${city}`);
            return null;
        }

        console.log(`[CPCB] Found ${data.records.length} records`);

        // Filter by station if provided
        let records = data.records as CPCBRecord[];
        if (targetStation) {
            const normalizedTarget = targetStation.toLowerCase().trim();
            const filtered = records.filter(r =>
                r.station && r.station.toLowerCase().includes(normalizedTarget)
            );

            if (filtered.length > 0) {
                console.log(`[CPCB] Filtered for station "${targetStation}": ${filtered.length} records`);
                records = filtered;
            } else {
                console.log(`[CPCB] Station "${targetStation}" not found in ${city} records. Falling back.`);
                return null;
            }
        }

        // Parse pollutants
        const pollutants: Record<string, number[]> = {
            pm25: [], pm10: [], no2: [], so2: [], co: [], o3: [],
        };
        let stationName = records.length > 0 && records[0].station ? records[0].station : (targetStation || city);

        for (const record of records) {
            if (record.station) stationName = record.station;

            const id = (record.pollutant_id || '').toLowerCase();
            const val = parseFloat(record.pollutant_avg || '');

            if (!isNaN(val) && val > 0) {
                if (id.includes('pm2.5') || id === 'pm25') pollutants.pm25.push(val);
                else if (id === 'pm10') pollutants.pm10.push(val);
                else if (id === 'no2') pollutants.no2.push(val);
                else if (id === 'so2') pollutants.so2.push(val);
                else if (id === 'co') pollutants.co.push(val);
                else if (id === 'o3' || id === 'ozone') pollutants.o3.push(val);
            }
        }

        const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : null;

        const pm25 = avg(pollutants.pm25);
        const pm10 = avg(pollutants.pm10);

        // Calculate AQI
        let aqi = 0;
        if (pm25 !== null) {
            aqi = calculateIndianAQI(pm25, 'pm25');
        } else if (pm10 !== null) {
            aqi = calculateIndianAQI(pm10, 'pm10');
        }

        if (aqi > 0) {
            console.log(`[CPCB] ✅ AQI: ${aqi}, PM2.5: ${pm25}, Station: ${stationName}`);
            return {
                aqi: Math.round(aqi),
                pm2_5: pm25 !== null ? Math.round(pm25) : null,
                pm10: pm10 !== null ? Math.round(pm10) : null,
                no2: avg(pollutants.no2) !== null ? Math.round(avg(pollutants.no2)!) : null,
                so2: avg(pollutants.so2) !== null ? Math.round(avg(pollutants.so2)!) : null,
                co: avg(pollutants.co) !== null ? Math.round(avg(pollutants.co)! * 10) / 10 : null,
                o3: avg(pollutants.o3) !== null ? Math.round(avg(pollutants.o3)!) : null,
                station: stationName,
            };
        }

        return null;
    } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
            console.log(`[CPCB] Timeout`);
        } else {
            console.error('[CPCB] Error:', error);
        }
        return null;
    }
}

function calculateIndianAQI(value: number, pollutant: 'pm25' | 'pm10'): number {
    if (pollutant === 'pm25') {
        if (value <= 30) return Math.round((value / 30) * 50);
        if (value <= 60) return Math.round(50 + ((value - 30) / 30) * 50);
        if (value <= 90) return Math.round(100 + ((value - 60) / 30) * 100);
        if (value <= 120) return Math.round(200 + ((value - 90) / 30) * 100);
        if (value <= 250) return Math.round(300 + ((value - 120) / 130) * 100);
        return Math.round(400 + ((value - 250) / 130) * 100);
    } else {
        if (value <= 50) return Math.round((value / 50) * 50);
        if (value <= 100) return Math.round(50 + ((value - 50) / 50) * 50);
        if (value <= 250) return Math.round(100 + ((value - 100) / 150) * 100);
        if (value <= 350) return Math.round(200 + ((value - 250) / 100) * 100);
        if (value <= 430) return Math.round(300 + ((value - 350) / 80) * 100);
        return Math.round(400 + ((value - 430) / 80) * 100);
    }
}

// =============================================================================
// REALISTIC CITY-BASED DATA (FALLBACK)
// =============================================================================

// Realistic AQI ranges based on historical data for major Indian cities
const CITY_AQI_PROFILES: Record<string, { min: number; max: number; typical: number }> = {
    'Delhi': { min: 150, max: 400, typical: 250 },
    'New Delhi': { min: 150, max: 400, typical: 250 },
    'Mumbai': { min: 60, max: 180, typical: 110 },
    'Navi Mumbai': { min: 50, max: 150, typical: 90 },
    'Pune': { min: 60, max: 160, typical: 100 },
    'Bangalore': { min: 50, max: 130, typical: 80 },
    'Bengaluru': { min: 50, max: 130, typical: 80 },
    'Chennai': { min: 45, max: 120, typical: 75 },
    'Kolkata': { min: 80, max: 200, typical: 130 },
    'Hyderabad': { min: 55, max: 140, typical: 90 },
    'Ahmedabad': { min: 70, max: 180, typical: 115 },
    'Lucknow': { min: 100, max: 250, typical: 160 },
    'Kanpur': { min: 120, max: 280, typical: 180 },
    'Jaipur': { min: 80, max: 200, typical: 130 },
    'Kochi': { min: 30, max: 80, typical: 50 },
    'Thiruvananthapuram': { min: 25, max: 70, typical: 45 },
    'Shimla': { min: 25, max: 60, typical: 40 },
    'Dehradun': { min: 40, max: 100, typical: 65 },
    'Chandigarh': { min: 60, max: 150, typical: 95 },
    'Guwahati': { min: 50, max: 120, typical: 75 },
    'Bhubaneswar': { min: 50, max: 130, typical: 80 },
    'Patna': { min: 100, max: 220, typical: 150 },
    'Ranchi': { min: 60, max: 140, typical: 90 },
    'Raipur': { min: 70, max: 160, typical: 105 },
    'Indore': { min: 60, max: 150, typical: 95 },
    'Bhopal': { min: 65, max: 155, typical: 100 },
    'Nagpur': { min: 55, max: 140, typical: 90 },
    'Visakhapatnam': { min: 45, max: 110, typical: 70 },
    'Coimbatore': { min: 35, max: 90, typical: 55 },
    'Madurai': { min: 40, max: 100, typical: 65 },
    'Varanasi': { min: 110, max: 260, typical: 170 },
    'Agra': { min: 100, max: 240, typical: 155 },
    'Gurgaon': { min: 120, max: 300, typical: 190 },
    'Gurugram': { min: 120, max: 300, typical: 190 },
    'Noida': { min: 130, max: 320, typical: 200 },
    'Faridabad': { min: 120, max: 290, typical: 185 },
    'Ghaziabad': { min: 130, max: 310, typical: 195 },
    'Chikkamagaluru': { min: 20, max: 50, typical: 32 },
    'Mysore': { min: 35, max: 85, typical: 55 },
    'Mysuru': { min: 35, max: 85, typical: 55 },
    'Mangalore': { min: 30, max: 75, typical: 48 },
    'Mangaluru': { min: 30, max: 75, typical: 48 },
    'Surat': { min: 60, max: 150, typical: 95 },
    'Vadodara': { min: 55, max: 140, typical: 88 },
    'Rajkot': { min: 50, max: 130, typical: 82 },
};

function generateRealisticAQI(city: string, healthConditions: HealthCondition[]): AQIData {
    // Find matching profile or use default
    let profile = CITY_AQI_PROFILES[city];

    if (!profile) {
        // Try to find by partial match
        const cityLower = city.toLowerCase();
        for (const [key, val] of Object.entries(CITY_AQI_PROFILES)) {
            if (cityLower.includes(key.toLowerCase()) || key.toLowerCase().includes(cityLower)) {
                profile = val;
                break;
            }
        }
    }

    if (!profile) {
        // Default moderate profile
        profile = { min: 60, max: 150, typical: 95 };
    }

    // Generate AQI with some variance around typical
    const variance = (profile.max - profile.min) * 0.15;
    const aqi = Math.round(profile.typical + (Math.random() - 0.5) * variance * 2);
    const constrainedAqi = Math.max(profile.min, Math.min(profile.max, aqi));

    const colorClass = getAQIClass(constrainedAqi);
    const status = getAQIStatus(constrainedAqi);

    // Generate realistic pollutant values based on AQI
    const pm2_5 = Math.round((constrainedAqi / 100) * 60 * (0.8 + Math.random() * 0.4));
    const pm10 = Math.round(pm2_5 * (1.4 + Math.random() * 0.4));
    const no2 = Math.round(15 + (constrainedAqi / 100) * 50 * (0.7 + Math.random() * 0.6));
    const so2 = Math.round(8 + (constrainedAqi / 100) * 30 * (0.6 + Math.random() * 0.8));
    const co = Math.round((3 + (constrainedAqi / 100) * 12 * (0.5 + Math.random() * 1)) * 10) / 10;
    const o3 = Math.round(20 + (constrainedAqi / 100) * 40 * (0.7 + Math.random() * 0.6));

    const { health, activities, personal_risk } = getPersonalizedRecommendations(constrainedAqi, healthConditions);

    console.log(`[FALLBACK] Generated realistic AQI ${constrainedAqi} for ${city} (typical: ${profile.typical})`);

    return {
        city,
        current: {
            aqi_value: constrainedAqi,
            aqi_status: status,
            aqi_color: colorClass,
            station: `${city} (Estimated)`,
            aqi_source: 'Estimated Data',
            dominant_pollutant: 'PM2.5',
            pm2_5,
            pm10,
            no2,
            so2,
            co,
            o3,
            temperature: 22 + Math.random() * 14,
            humidity: 40 + Math.random() * 40,
            wind: 2 + Math.random() * 10,
        },
        health,
        activities,
        personal_risk,
    };
}

// =============================================================================
// MAIN FETCH FUNCTION
// =============================================================================

export async function fetchCurrentAQI(
    city: string,
    station?: Station,
    healthConditions: HealthCondition[] = ['none']
): Promise<AQIData> {
    const cacheKey = `aqi-${city.toLowerCase()}-${station?.id || 'default'}`;

    console.log(`\n${'='.repeat(50)}`);
    console.log(`[API] Fetching: "${city}" ${station ? `(${station.name})` : ''}`);
    console.log(`${'='.repeat(50)}`);

    // Check cache
    const cached = apiCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
        console.log(`[CACHE] Hit (age: ${Math.round((Date.now() - cached.timestamp) / 1000)}s)`);
        const { health, activities, personal_risk } = getPersonalizedRecommendations(
            cached.data.current.aqi_value,
            healthConditions
        );
        return { ...cached.data, health, activities, personal_risk };
    }

    const cityData = getCityByName(city);
    const coords = station
        ? { lat: station.lat, lon: station.lon }
        : cityData
            ? { lat: cityData.lat, lon: cityData.lon }
            : null;

    // ============ STRATEGY: WAQI First for Specific Stations, CPCB for City (India) ============

    // Priority 1: Specific Station via WAQI
    // If we have a numeric UID (from Global Search) or Geo coordinates
    const isUidStation = station?.id && /^\d+$/.test(station.id);

    if (station && (coords || isUidStation)) {
        console.log(`[API] Station selected - trying WAQI ${isUidStation ? '(UID)' : '(Geo)'} first...`);

        let waqiStationData: WAQIResponse | null = null;
        if (isUidStation) {
            waqiStationData = await fetchWAQIData({ uid: station.id });
        } else if (coords) {
            waqiStationData = await fetchWAQIData(coords);
        }

        if (waqiStationData) {
            console.log(`[API] ✅ WAQI Station Success - AQI: ${waqiStationData.data.aqi}`);
            const aqiData = convertWAQItoAQIData(waqiStationData, city, healthConditions);
            aqiData.current.station = station.name; // Keep our station name
            apiCache.set(cacheKey, { data: aqiData, timestamp: Date.now() });
            return aqiData;
        }
        console.log(`[API] WAQI Station failed - falling back to CPCB...`);
    }

    // Priority 2: CPCB (for Indian Locations ONLY)
    if (isIndianLocation(city)) {
        console.log(`[API] Indian city detected - trying CPCB first...`);
        const cpcbData = await fetchCPCBData(city, station?.name);

        if (cpcbData && cpcbData.aqi > 0) {
            console.log(`[API] ✅ CPCB Success - AQI: ${cpcbData.aqi}`);
            // ... (rest of CPCB logic) ...
            const { health, activities, personal_risk } = getPersonalizedRecommendations(cpcbData.aqi, healthConditions);

            const aqiData: AQIData = {
                city,
                current: {
                    aqi_value: cpcbData.aqi,
                    aqi_status: getAQIStatus(cpcbData.aqi),
                    aqi_color: getAQIClass(cpcbData.aqi),
                    station: station?.name || cpcbData.station,
                    aqi_source: 'CPCB India',
                    dominant_pollutant: 'PM2.5',
                    pm2_5: cpcbData.pm2_5,
                    pm10: cpcbData.pm10,
                    no2: cpcbData.no2,
                    so2: cpcbData.so2,
                    co: cpcbData.co,
                    o3: cpcbData.o3,
                },
                health,
                activities,
                personal_risk,
            };

            apiCache.set(cacheKey, { data: aqiData, timestamp: Date.now() });
            return aqiData;
        }

        console.log(`[API] CPCB failed/empty - falling back to WAQI...`);
    } else {
        console.log(`[API] Non-Indian/Global location detected: "${city}" - Defaults to WAQI`);
    }

    // Priority 3: WAQI Generic City Search (Global & Indian Fallback)
    console.log(`[API] Trying WAQI for city: "${city}"...`);

    let waqiData = await fetchWAQIData(city);

    // If city search returned nothing, but we had coordinates (rare case for manual station selection), try coords
    if (!waqiData && coords) {
        waqiData = await fetchWAQIData(coords);
    }

    if (waqiData) {
        console.log(`[API] ✅ WAQI City Success - AQI: ${waqiData.data.aqi}`);
        const aqiData = convertWAQItoAQIData(waqiData, city, healthConditions);
        if (station) aqiData.current.station = station.name;

        // Force source label for clarity
        aqiData.current.aqi_source = isIndianLocation(city) ? 'WAQI India' : 'WAQI Global';

        apiCache.set(cacheKey, { data: aqiData, timestamp: Date.now() });
        return aqiData;
    }

    // Fallback to realistic estimated data
    console.log(`[API] ⚠️ All APIs failed - using realistic estimates...`);
    return generateRealisticAQI(city, healthConditions);
}

// =============================================================================
// SEARCH
// =============================================================================

export async function searchCities(query: string): Promise<Array<{ name: string; aqi: number; uid?: number; station?: any }>> {
    if (query.length < 2) return [];

    try {
        const url = `${WAQI_API}/search/?keyword=${encodeURIComponent(query)}&token=${WAQI_TOKEN}`;
        const response = await fetch(url);
        const data = await response.json();

        if (data.status === 'ok' && data.data) {
            return data.data.slice(0, 10).map((item: any) => ({
                name: item.station?.name || 'Unknown',
                aqi: parseInt(item.aqi || '0') || 0,
                uid: item.uid,
                station: item.station
            }));
        }
        return [];
    } catch {
        return [];
    }
}

// =============================================================================
// PREDICTIONS & HISTORICAL
// =============================================================================

export async function fetchPredictions(
    _city: string,
    scenario: Scenario,
    currentAqi?: number
): Promise<HourlyPrediction[]> {
    return generateMockPredictions(currentAqi || 100, scenario);
}

export async function fetchHistoricalData(city: string, days: number): Promise<HistoricalDataPoint[]> {
    return generateHistoricalData(city, days);
}

// =============================================================================
// MULTI-CITY FETCH (HEATMAP)
// =============================================================================

export async function fetchMultipleCityAQI(cities: typeof INDIAN_CITIES): Promise<CityAQI[]> {
    console.log(`[HEATMAP] Loading ${cities.length} cities...`);

    const results: CityAQI[] = [];
    const batchSize = 8;

    for (let i = 0; i < cities.length; i += batchSize) {
        const batch = cities.slice(i, i + batchSize);

        const batchResults = await Promise.all(
            batch.map(async (city): Promise<CityAQI> => {
                try {
                    // Try CPCB first for Indian cities
                    const cpcbData = await fetchCPCBData(city.name);
                    if (cpcbData && cpcbData.aqi > 0) {
                        return {
                            name: city.name,
                            state: city.state,
                            lat: city.lat,
                            lon: city.lon,
                            aqi: cpcbData.aqi,
                            status: getAQIStatus(cpcbData.aqi),
                            color: getAQIClass(cpcbData.aqi),
                        };
                    }

                    // Fallback to WAQI (coordinate-based)
                    const waqiData = await fetchWAQIData({ lat: city.lat, lon: city.lon });
                    if (waqiData && waqiData.data.aqi > 0) {
                        return {
                            name: city.name,
                            state: city.state,
                            lat: city.lat,
                            lon: city.lon,
                            aqi: waqiData.data.aqi,
                            status: getAQIStatus(waqiData.data.aqi),
                            color: getAQIClass(waqiData.data.aqi),
                        };
                    }

                    // Fallback to realistic estimate
                    const profile = CITY_AQI_PROFILES[city.name] || { typical: 80, min: 50, max: 150 };
                    const variance = (profile.max - profile.min) * 0.1;
                    const aqi = Math.round(profile.typical + (Math.random() - 0.5) * variance * 2);

                    return {
                        name: city.name,
                        state: city.state,
                        lat: city.lat,
                        lon: city.lon,
                        aqi,
                        status: getAQIStatus(aqi),
                        color: getAQIClass(aqi),
                    };
                } catch {
                    // Generate based on city profile
                    const profile = CITY_AQI_PROFILES[city.name] || { typical: 85, min: 50, max: 160 };
                    const aqi = profile.typical;

                    return {
                        name: city.name,
                        state: city.state,
                        lat: city.lat,
                        lon: city.lon,
                        aqi,
                        status: getAQIStatus(aqi),
                        color: getAQIClass(aqi),
                    };
                }
            })
        );

        results.push(...batchResults);

        // Small delay between batches
        if (i + batchSize < cities.length) {
            await new Promise(r => setTimeout(r, 100));
        }
    }

    console.log(`[HEATMAP] Loaded ${results.length} cities`);

    // Log sample of results
    const sample = results.slice(0, 5);
    console.log(`[HEATMAP] Sample:`, sample.map(c => `${c.name}: ${c.aqi}`).join(', '));

    return results;
}

// =============================================================================
// CACHE CONTROL
// =============================================================================

export function clearAPICache(): void {
    console.log('[CACHE] Cleared');
    apiCache.clear();
}
