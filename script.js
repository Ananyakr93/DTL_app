const API_BASE = "";  // Use relative URLs for deployment compatibility

let currentCity = "Bangalore";
let refreshInterval;
let countdownInterval;
let remainingSeconds = 300; // 5 minutes

// Initialize Socket.IO (optional - only if available)
let socket = null;
try {
    if (typeof io !== 'undefined') {
        socket = io(window.location.origin);

        socket.on('connect', () => {
            console.log("CONNECTED to WebSocket! ⚡");
            if (currentCity) {
                socket.emit('join', { city: currentCity });
            }
        });

        socket.on('aqi_update', (data) => {
            console.log("🔴 Real-time update received!", data);
            updateUI(data);
        });
    } else {
        console.log("ℹ️ Socket.IO not available - using polling mode");
    }
} catch (e) {
    console.log("ℹ️ WebSocket not available:", e.message);
}

// Separate UI update logic from fetch logic
function updateUI(data) {
    const aqiCard = document.getElementById("aqiCard");
    aqiCard.className = `card ${data.current.aqi_color}`;
    document.getElementById("aqiValue").innerText = data.current.aqi_value;
    document.getElementById("aqiStatus").innerText = data.current.aqi_status;

    // Update Pollutants
    document.getElementById("pm25").innerText = data.current.pm2_5;
    document.getElementById("pm10").innerText = data.current.pm10;
    document.getElementById("no2").innerText = data.current.no2;
    document.getElementById("so2").innerText = data.current.so2;
    document.getElementById("o3").innerText = data.current.o3;
    document.getElementById("co").innerText = data.current.co;

    // Update Source
    document.getElementById("dataSource").innerText = data.current.source.toUpperCase();

    // Updated Time
    const now = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    document.getElementById("updateTime").innerText = `📍 ${data.city} | Live Update: ${now}`;
}

/* ================= UTILITY FUNCTIONS ================= */
function showLoading() {
    document.getElementById("loadingIndicator").style.display = "flex";
    document.getElementById("mainContent").style.display = "none";
}

function hideLoading() {
    document.getElementById("loadingIndicator").style.display = "none";
    document.getElementById("mainContent").style.display = "block";
}

function showError(message) {
    alert(`Error: ${message}`);
    hideLoading();
}

function getAQIClass(aqi) {
    if (aqi <= 50) return "good";
    if (aqi <= 100) return "moderate";
    if (aqi <= 200) return "poor";
    if (aqi <= 300) return "unhealthy";
    return "severe";
}

function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

/* ================= CURRENT AQI ================= */
async function loadCurrentAQI(lat = null, lon = null) {
    try {
        showLoading();

        let url = `${API_BASE}/api/current?city=${encodeURIComponent(currentCity)}`;
        if (lat && lon) {
            url = `${API_BASE}/api/current?lat=${lat}&lon=${lon}`;
        }

        const res = await fetch(url);

        if (!res.ok) {
            throw new Error("Failed to fetch data");
        }

        const data = await res.json();



        // Update current city if we found it via coordinates
        if (data.city && data.city !== "Current Location") {
            currentCity = data.city;
            const searchInput = document.getElementById("citySearch");
            if (searchInput && searchInput.value !== currentCity) {
                searchInput.value = currentCity;
            }

            // Join WebSocket room for this city
            if (socket) {
                socket.emit('join', { city: currentCity });
            }
        }

        // Reuse UI update logic
        updateUI(data);

        // Update Health Recommendations
        if (data.health) {
            document.getElementById("healthGeneral").innerText = data.health.general;
            document.getElementById("healthSensitive").innerText = data.health.sensitive;
        }

        // Update Activity Cards
        const activityBox = document.getElementById("activityCards");
        activityBox.innerHTML = "";

        if (data.activities && data.activities.length > 0) {
            data.activities.forEach(act => {
                const div = document.createElement("div");
                div.className = `activity-card ${data.current.aqi_value <= 100 ? 'allow' : 'avoid'}`;
                div.innerHTML = `
                    <span class="activity-icon">${data.current.aqi_value <= 100 ? '✅' : '⚠️'}</span>
                    <span class="activity-text">${act}</span>
                `;
                activityBox.appendChild(div);
            });
        }

        hideLoading();

        // Reset countdown
        remainingSeconds = 300;

    } catch (err) {
        console.error("Error loading current AQI:", err);
        showError(err.message || "Failed to load current data");
    }
}

/* ================= AQI PREDICTION (24 HOURS) ================= */
async function loadPrediction() {
    try {
        const scenario = document.getElementById("scenarioSelect")?.value || "normal";
        const res = await fetch(`${API_BASE}/api/predict?city=${encodeURIComponent(currentCity)}&hours=24&scenario=${scenario}`);

        if (!res.ok) {
            throw new Error("Failed to fetch prediction");
        }

        const data = await res.json();

        const box = document.getElementById("predictionBars");
        box.innerHTML = "";

        if (!data || data.length === 0) {
            box.innerHTML = '<p class="no-data">No prediction data available</p>';
            return;
        }

        // Find max AQI for scaling
        const maxAQI = Math.max(...data.map(p => p.aqi_upper || p.aqi));
        const scaleFactor = 200 / Math.max(maxAQI, 100);

        data.forEach((p, index) => {
            const wrap = document.createElement("div");
            wrap.className = "pred-wrapper";

            // Main AQI value
            const val = document.createElement("div");
            val.className = "bar-value";
            val.innerText = p.aqi;

            // Uncertainty range if available
            if (p.aqi_lower !== undefined && p.aqi_upper !== undefined) {
                const uncertainty = document.createElement("div");
                uncertainty.className = "bar-uncertainty";
                uncertainty.innerText = `±${p.uncertainty || Math.round((p.aqi_upper - p.aqi_lower) / 2)}`;
                val.appendChild(document.createElement("br"));
                val.appendChild(uncertainty);
            }

            // Main bar with uncertainty shading
            const barContainer = document.createElement("div");
            barContainer.className = "bar-container";

            // Uncertainty band (background)
            if (p.aqi_lower !== undefined && p.aqi_upper !== undefined) {
                const uncertaintyBand = document.createElement("div");
                uncertaintyBand.className = "uncertainty-band";
                const bandHeight = (p.aqi_upper - p.aqi_lower) * scaleFactor;
                const bandBottom = p.aqi_lower * scaleFactor;
                uncertaintyBand.style.height = bandHeight + "px";
                uncertaintyBand.style.bottom = bandBottom + "px";
                barContainer.appendChild(uncertaintyBand);
            }

            // Main bar
            const bar = document.createElement("div");
            bar.className = `bar ${p.color}`;
            bar.style.height = Math.max(40, p.aqi * scaleFactor) + "px";
            bar.title = `${p.hour}: AQI ${p.aqi} (${p.status})${p.aqi_lower ? ` [${p.aqi_lower}-${p.aqi_upper}]` : ''}`;
            barContainer.appendChild(bar);

            const label = document.createElement("div");
            label.className = "bar-label";
            label.innerText = p.hour;

            const status = document.createElement("div");
            status.className = "bar-status";
            status.innerText = p.status.substring(0, 4);

            wrap.appendChild(val);
            wrap.appendChild(barContainer);
            wrap.appendChild(label);
            wrap.appendChild(status);
            box.appendChild(wrap);
        });

    } catch (err) {
        console.error("Error loading prediction:", err);
        document.getElementById("predictionBars").innerHTML =
            '<p class="error-message">Failed to load prediction data</p>';
    }
}

/* ================= CITY SEARCH ================= */
function searchCity() {
    const input = document.getElementById("citySearch");
    const city = input.value.trim();

    if (city !== "" && city !== currentCity) {
        currentCity = city;
        loadAllData();
    }
}

/* ================= AUTO LOCATION & SUGGESTIONS ================= */
function detectLocation() {
    if (!navigator.geolocation) {
        alert("Geolocation is not supported by your browser");
        return;
    }

    showLoading();
    navigator.geolocation.getCurrentPosition(
        (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            loadAllData(lat, lon);
        },
        (error) => {
            console.error("Geolocation error:", error);
            hideLoading();
            alert("Unable to retrieve your location. Using default city.");
            loadAllData(); // Fallback
        }
    );
}

async function fetchSuggestions(query) {
    if (query.length < 2) return;

    try {
        const res = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(query)}`);
        const suggestions = await res.json();

        const datalist = document.getElementById("city-suggestions");
        datalist.innerHTML = "";

        suggestions.forEach(s => {
            const option = document.createElement("option");
            option.value = s.name; // Browser auto-fills this
            option.label = s.display_name;
            datalist.appendChild(option);
        });
    } catch (e) {
        console.error("Error fetching suggestions:", e);
    }
}

const handleInput = debounce((e) => {
    fetchSuggestions(e.target.value);
}, 300);

// Search on Enter key and Inputs
document.addEventListener("DOMContentLoaded", () => {
    const searchInput = document.getElementById("citySearch");

    if (searchInput) {
        searchInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") {
                searchCity();
            }
        });

        searchInput.addEventListener("input", handleInput);
    }
});

/* ================= AUTO REFRESH & COUNTDOWN ================= */
function startCountdown() {
    if (countdownInterval) {
        clearInterval(countdownInterval);
    }

    countdownInterval = setInterval(() => {
        remainingSeconds--;

        const minutes = Math.floor(remainingSeconds / 60);
        const seconds = remainingSeconds % 60;

        const timerEl = document.getElementById("refreshTimer");
        if (timerEl) {
            timerEl.innerText = `Auto-refresh: ${minutes}:${seconds.toString().padStart(2, '0')}`;
        }

        if (remainingSeconds <= 0) {
            remainingSeconds = 300;
            loadAllData();
        }
    }, 1000);
}

function startAutoRefresh() {
    // Clear existing interval
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }

    // Refresh every 5 minutes
    refreshInterval = setInterval(() => {
        loadAllData();
    }, 300000);

    // Start countdown timer
    startCountdown();
}

/* ================= LOAD ALL DATA ================= */
async function loadAllData(lat = null, lon = null) {
    await loadCurrentAQI(lat, lon);
    await loadPrediction();
}

/* ================= INITIAL LOAD ================= */
document.addEventListener("DOMContentLoaded", () => {
    // Load saved city preference
    const savedCity = localStorage.getItem('defaultCity');
    if (savedCity) {
        currentCity = savedCity;
        const searchInput = document.getElementById("citySearch");
        if (searchInput) searchInput.value = savedCity;
    }

    // Load saved city preference via logic or Auto-Detect
    // If no specific previous city is crucial, we try auto-detect first
    detectLocation();
    // Only if auto-detect fails or is slow, the logic inside detectLocation handles fallback or we can set a timeout
    // For now, detectLocation calls loadAllData() on success/fail.
    startAutoRefresh();
});

/* ================= CLEANUP ================= */
window.addEventListener("beforeunload", () => {
    if (refreshInterval) clearInterval(refreshInterval);
    if (countdownInterval) clearInterval(countdownInterval);
});