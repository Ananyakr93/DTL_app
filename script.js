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
            // console.log("CONNECTED to WebSocket! ⚡");
            if (currentCity) {
                socket.emit('join', { city: currentCity });
            }
        });

        socket.on('aqi_update', (data) => {
            // console.log("🔴 Real-time update received!", data);
            updateUI(data);
        });
    } else {
        // console.log("ℹ️ Socket.IO not available - using polling mode");
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

function showPersonalAlert(risk) {
    let alertBox = document.getElementById('personalAlertBox');

    // Create if missing
    if (!alertBox) {
        alertBox = document.createElement('div');
        alertBox.id = 'personalAlertBox';
        // Insert after data source badge
        const badge = document.getElementById('dataSourceBadge');
        if (badge && badge.parentNode) {
            badge.parentNode.insertBefore(alertBox, badge.nextSibling);
        } else {
            const mc = document.getElementById('mainContent');
            mc.prepend(alertBox);
        }

        // Base Styles
        alertBox.style.marginTop = "15px";
        alertBox.style.marginBottom = "15px";
        alertBox.style.padding = "15px";
        alertBox.style.borderRadius = "12px";
        alertBox.style.display = "none";
        alertBox.style.animation = "fadeIn 0.5s ease-in-out";
    }

    // Logic: Show only if Moderate+ for sensitive or Poor+ for normal
    // The backend provides 'risk_level'

    if (risk.risk_level === 'High' || risk.risk_level === 'Severe' || (risk.alert_message && risk.risk_level !== 'Low')) {
        alertBox.style.display = 'block';
        const isSevere = risk.risk_level === 'Severe';
        const isHigh = risk.risk_level === 'High';

        alertBox.style.backgroundColor = isSevere ? '#ffebee' : (isHigh ? '#fff3e0' : '#e3f2fd');
        alertBox.style.borderLeft = isSevere ? '5px solid #ef5350' : (isHigh ? '5px solid #ff9800' : '5px solid #2196f3');
        alertBox.style.color = '#333';

        alertBox.innerHTML = `
            <div style="display:flex; flex-direction:column; gap:8px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:24px">${isSevere ? '🚨' : (isHigh ? '⚠️' : 'ℹ️')}</span>
                    <div>
                        <h3 style="margin:0; font-size:16px; font-weight:bold; color:${isSevere ? '#c62828' : '#e65100'}">
                            ${risk.alert_title || 'Health Update'}
                        </h3>
                        <p style="margin:2px 0 0 0; font-size:14px;">${risk.alert_message}</p>
                    </div>
                </div>
                ${risk.tips && risk.tips.length > 0 ? `
                <div style="background:rgba(255,255,255,0.5); padding:10px; border-radius:8px;">
                    <strong style="font-size:12px; text-transform:uppercase; color:#666;">Recommended Actions:</strong>
                    <ul style="margin:5px 0 0 20px; padding:0; font-size:14px;">
                        ${risk.tips.map(t => `<li>${t}</li>`).join('')}
                    </ul>
                </div>` : ''}
            </div>
        `;
    } else {
        alertBox.style.display = 'none';
        alertBox.innerHTML = '';
    }
}

/* ================= CURRENT AQI ================= */
async function loadCurrentAQI(lat = null, lon = null) {
    try {
        showLoading();

        const healthMode = localStorage.getItem('aeroCleanHealthMode') || 'normal';
        let url = `${API_BASE}/api/current?city=${encodeURIComponent(currentCity)}&health_mode=${healthMode}`;
        if (lat && lon) {
            url = `${API_BASE}/api/current?lat=${lat}&lon=${lon}&health_mode=${healthMode}`;
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

        // Show Personal Alert
        if (data.personal_risk) {
            showPersonalAlert(data.personal_risk);
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
        const healthMode = localStorage.getItem('aeroCleanHealthMode') || 'normal';
        const res = await fetch(`${API_BASE}/api/predict?city=${encodeURIComponent(currentCity)}&hours=24&scenario=${scenario}&health_mode=${healthMode}`);

        if (!res.ok) {
            throw new Error("Failed to fetch prediction");
        }


        const jsonResponse = await res.json();
        const data = jsonResponse.predictions || [];

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
        localStorage.setItem('selectedCity', city); // Persist for other pages
        localStorage.setItem('defaultCity', city);  // Save preference
        loadAllData();
    }
}

/* ================= AUTO LOCATION & SUGGESTIONS ================= */
function detectLocation() {
    if (!navigator.geolocation) {
        console.warn("Geolocation not supported");
        loadAllData(); // Fallback immediately
        return;
    }

    showLoading();

    let locationFound = false;

    // Set a timeout of 3 seconds
    const timeoutId = setTimeout(() => {
        if (!locationFound) {
            console.warn("Location request timed out. Loading default.");
            locationFound = true; // Prevent late success from overriding
            loadAllData(); // Fallback to default city
        }
    }, 3000);

    navigator.geolocation.getCurrentPosition(
        (position) => {
            if (!locationFound) {
                locationFound = true;
                clearTimeout(timeoutId);
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                loadAllData(lat, lon);
            }
        },
        (error) => {
            if (!locationFound) {
                locationFound = true;
                clearTimeout(timeoutId);
                console.error("Geolocation error:", error);
                // Don't alert() on startup, just fallback
                loadAllData();
            }
        },
        { timeout: 3000, maximumAge: 600000 }
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
    // Load saved city preference (Shared across pages)
    const savedCity = localStorage.getItem('selectedCity') || localStorage.getItem('defaultCity');
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