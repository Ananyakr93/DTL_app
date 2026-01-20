from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("⚠️ flask-socketio not installed - WebSocket features disabled")
import threading
import time
# ... existing imports ...

# ... existing imports ...


from flask_caching import Cache
import requests
import numpy as np
from datetime import datetime, timedelta
import os
import sqlite3
from sklearn.ensemble import IsolationForest

# Optional TensorFlow import
try:
    from tensorflow.keras.models import load_model
    import joblib
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    joblib = None
    print("⚠️ TensorFlow not installed - LSTM predictions disabled")


# ===================== APP SETUP =====================
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
# Disable WebSockets on Vercel (serverless doesn't support persistent connections)
if SOCKETIO_AVAILABLE and os.environ.get('VERCEL') != '1':
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
else:
    socketio = None

# ===================== WEB ROUTES =====================
@app.route("/")
def index():
    """Serve the dashboard"""
    return send_from_directory('.', 'index.html')

@app.route("/<path:path>")
def static_files(path):
    """Serve other static files"""
    return send_from_directory('.', path)

# Flask-Caching configuration
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
})

# ===================== API CONFIG =====================
# AQICN API for real-time current AQI (accurate India data)
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "91cfb794c918bbc8fed384ff6aab22383dec190a")  # Replace 'demo' with actual token
AQICN_URL = "https://api.waqi.info/feed"

# Open-Meteo API for predictions (free, no key needed)
OPENMETEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Geocoding API (free)
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# ===================== DATABASE SETUP =====================
# On Vercel, use /tmp directory (ephemeral storage)
IS_VERCEL = os.environ.get('VERCEL') == '1'
if IS_VERCEL:
    DB_PATH = '/tmp/aeroclean.db'
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), "aeroclean.db")

def init_db():
    """Initialize SQLite database for user profiles"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        name TEXT,
        health_conditions TEXT,
        alert_threshold INTEGER DEFAULT 100,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        gdpr_consent INTEGER DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS anomaly_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        aqi_value INTEGER,
        anomaly_type TEXT,
        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        root_cause TEXT
    )''')
    # New: Prediction logs for validation
    c.execute('''CREATE TABLE IF NOT EXISTS prediction_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        forecast_hour INTEGER,
        predicted_aqi INTEGER,
        actual_aqi INTEGER,
        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        validation_time TIMESTAMP,
        error INTEGER
    )''')
    conn.commit()
    conn.close()

init_db()

# ===================== LAZY LOAD IMPROVED LSTM MODEL =====================
# Model is loaded lazily on first prediction request to avoid blocking startup
MODEL_LOADED = False
model = None
scaler = None
config = {'lookback': 48, 'forecast_horizon': 24, 'n_features': 1}
_model_loading = False  # Prevents concurrent load attempts

def get_model():
    """Lazy load the LSTM model on first use"""
    global MODEL_LOADED, model, scaler, config, _model_loading
    
    if MODEL_LOADED:
        return model, scaler, config
    
    if not TF_AVAILABLE:
        print("⚠️ TensorFlow not available - using API-based predictions only")
        return None, None, config
    
    if _model_loading:
        # Another thread is loading, wait briefly
        import time
        time.sleep(0.5)
        return model, scaler, config
    
    _model_loading = True
    try:
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.saving import register_keras_serializable

        @register_keras_serializable()
        class MCDropout(Dropout):
            """Monte Carlo Dropout - stays active during inference for uncertainty estimation"""
            def call(self, inputs):
                return super().call(inputs, training=True)

        model = load_model("aqi_lstm_model_improved.keras", custom_objects={"MCDropout": MCDropout})
        scaler = joblib.load("scaler_improved.pkl")
        config = joblib.load("model_config.pkl")
        MODEL_LOADED = True
        print(f"✅ Model loaded successfully!")
        print(f"   MAE: {config['mae']:.2f}, RMSE: {config['rmse']:.2f}")
    except Exception as e:
        print(f"⚠️ Model not found: {e}")
        print("   Using fallback prediction methods")
    finally:
        _model_loading = False
    
    return model, scaler, config

# ===================== HELPER FUNCTIONS =====================
def get_aqi_status(aqi):
    """Get AQI status based on Indian CPCB standards"""
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def get_aqi_color(aqi):
    """Get color class for AQI value"""
    if aqi <= 50: return "good"
    elif aqi <= 100: return "moderate"
    elif aqi <= 200: return "poor"
    elif aqi <= 300: return "unhealthy"
    else: return "severe"

def get_activity_recommendations(aqi):
    """Get activity recommendations based on AQI"""
    if aqi <= 50:
        return ["Perfect for outdoor exercise", "Morning walks recommended", 
                "Cycling and sports activities", "All outdoor activities safe"]
    elif aqi <= 100:
        return ["Light outdoor activities OK", "Short walks acceptable",
                "Consider indoor exercise", "Sensitive groups take precautions"]
    elif aqi <= 200:
        return ["Limit outdoor exposure", "Wear mask if going out",
                "Indoor activities preferred", "Avoid prolonged exertion"]
    elif aqi <= 300:
        return ["Avoid outdoor activities", "Use air purifier indoors",
                "Keep windows closed", "Mandatory masks for sensitive groups"]
    else:
        return ["Stay indoors strictly", "Use N95 mask if necessary",
                "Air purifier essential", "Avoid all outdoor exposure"]

def get_health_recommendations(aqi, health_conditions=None):
    """Get health recommendations with personalization for health profiles"""
    base_recommendations = {
        "general": "",
        "sensitive": ""
    }
    
    if aqi <= 50:
        base_recommendations["general"] = "Air quality is good. Enjoy outdoor activities!"
        base_recommendations["sensitive"] = "No health concerns for sensitive groups."
    elif aqi <= 100:
        base_recommendations["general"] = "Air quality is acceptable for most people."
        base_recommendations["sensitive"] = "Unusually sensitive people should consider reducing prolonged outdoor exertion."
    elif aqi <= 200:
        base_recommendations["general"] = "People with respiratory conditions should limit outdoor exertion."
        base_recommendations["sensitive"] = "Children, elderly, and people with lung/heart disease should avoid prolonged outdoor activities."
    elif aqi <= 300:
        base_recommendations["general"] = "Everyone should reduce outdoor exertion."
        base_recommendations["sensitive"] = "Sensitive groups should avoid all outdoor activities."
    else:
        base_recommendations["general"] = "Health alert: Everyone should avoid outdoor activities."
        base_recommendations["sensitive"] = "Sensitive groups must remain indoors with air purifiers."
    
    # Add personalized recommendations based on health conditions
    if health_conditions:
        conditions = health_conditions.split(',') if isinstance(health_conditions, str) else health_conditions
        personalized = []
        if 'asthma' in conditions and aqi > 100:
            personalized.append("⚠️ Asthma: Keep rescue inhaler nearby, avoid triggers")
        if 'heart_disease' in conditions and aqi > 150:
            personalized.append("⚠️ Heart condition: Avoid physical exertion, monitor symptoms")
        if 'elderly' in conditions and aqi > 100:
            personalized.append("⚠️ Elderly: Stay indoors, ensure good ventilation")
        if 'children' in conditions and aqi > 100:
            personalized.append("⚠️ Children: Keep indoors, avoid outdoor play")
        
        if personalized:
            base_recommendations["personalized"] = personalized
    
    return base_recommendations

def calculate_indian_aqi_pm25(pm25):
    """Calculate Indian AQI from PM2.5 concentration (CPCB standards)"""
    if pm25 <= 30: return int((pm25 / 30) * 50)
    elif pm25 <= 60: return int(51 + (pm25 - 30) * 49 / 30)
    elif pm25 <= 90: return int(101 + (pm25 - 60) * 99 / 30)
    elif pm25 <= 120: return int(201 + (pm25 - 90) * 99 / 30)
    elif pm25 <= 250: return int(301 + (pm25 - 120) * 99 / 130)
    else: return 500

def calculate_indian_aqi_pm10(pm10):
    """Calculate Indian AQI from PM10 concentration"""
    if pm10 <= 50: return int((pm10 / 50) * 50)
    elif pm10 <= 100: return int(51 + (pm10 - 50) * 49 / 50)
    elif pm10 <= 250: return int(101 + (pm10 - 100) * 99 / 150)
    elif pm10 <= 350: return int(201 + (pm10 - 250) * 99 / 100)
    elif pm10 <= 430: return int(301 + (pm10 - 350) * 99 / 80)
    else: return 500

# ===================== GEOCODING =====================
@cache.memoize(timeout=86400)  # Cache for 24 hours
def get_coordinates(city):
    """Get latitude and longitude for a city using Open-Meteo geocoding"""
    try:
        # Add India bias for Indian cities
        search_query = f"{city}, India" if city.lower() in [
            'bangalore', 'delhi', 'mumbai', 'chennai', 'kolkata', 'hyderabad',
            'pune', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur',
            'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri-chinchwad',
            'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik'
        ] else city
        
        res = requests.get(GEOCODING_URL, params={
            "name": search_query,
            "count": 1,
            "language": "en",
            "format": "json"
        }, timeout=10)
        
        data = res.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return {
                "lat": result["latitude"],
                "lon": result["longitude"],
                "name": result.get("name", city),
                "country": result.get("country", "")
            }
        raise ValueError(f"City '{city}' not found")
    except Exception as e:
        raise ValueError(f"Geocoding error: {str(e)}")

# ===================== AQICN API (Real-time Current AQI) =====================
@cache.memoize(timeout=300)  # Cache for 5 minutes
def fetch_aqicn_current(city):
    """Fetch real-time current AQI from AQICN API"""
    try:
        # Try city name directly
        url = f"{AQICN_URL}/{city}/?token={AQICN_TOKEN}"
        res = requests.get(url, timeout=10)
        data = res.json()
        
        if data.get("status") == "ok" and "data" in data:
            aqi_data = data["data"]
            iaqi = aqi_data.get("iaqi", {})
            
            return {
                "aqi_value": aqi_data.get("aqi", 0),
                "pm2_5": iaqi.get("pm25", {}).get("v", 0),
                "pm10": iaqi.get("pm10", {}).get("v", 0),
                "no2": iaqi.get("no2", {}).get("v", 0),
                "so2": iaqi.get("so2", {}).get("v", 0),
                "co": iaqi.get("co", {}).get("v", 0),
                "o3": iaqi.get("o3", {}).get("v", 0),
                "source": "aqicn",
                "station": aqi_data.get("city", {}).get("name", city)
            }
        return None
    except Exception as e:
        print(f"AQICN API error: {e}")
        return None

def fetch_aqicn_by_coords(lat, lon):
    """Fetch AQICN data by coordinates"""
    try:
        url = f"{AQICN_URL}/geo:{lat};{lon}/?token={AQICN_TOKEN}"
        res = requests.get(url, timeout=10)
        data = res.json()
        
        if data.get("status") == "ok" and "data" in data:
            aqi_data = data["data"]
            iaqi = aqi_data.get("iaqi", {})
            
            return {
                "aqi_value": aqi_data.get("aqi", 0),
                "pm2_5": iaqi.get("pm25", {}).get("v", 0),
                "pm10": iaqi.get("pm10", {}).get("v", 0),
                "no2": iaqi.get("no2", {}).get("v", 0),
                "so2": iaqi.get("so2", {}).get("v", 0),
                "co": iaqi.get("co", {}).get("v", 0),
                "o3": iaqi.get("o3", {}).get("v", 0),
                "source": "aqicn"
            }
        return None
    except:
        return None

# ===================== OPEN-METEO API (Predictions) =====================
@cache.memoize(timeout=3600)  # Cache for 1 hour
def fetch_openmeteo_forecast(lat, lon, hours=24):
    """Fetch air quality forecast from Open-Meteo API"""
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", 
                      "ozone", "carbon_monoxide", "european_aqi"],
            "forecast_days": 2,
            "timezone": "auto"
        }
        
        res = requests.get(OPENMETEO_URL, params=params, timeout=15)
        data = res.json()
        
        if "hourly" not in data:
            return None
        
        hourly = data["hourly"]
        times = hourly.get("time", [])
        pm25_values = hourly.get("pm2_5", [])
        pm10_values = hourly.get("pm10", [])
        no2_values = hourly.get("nitrogen_dioxide", [])
        so2_values = hourly.get("sulphur_dioxide", [])
        o3_values = hourly.get("ozone", [])
        co_values = hourly.get("carbon_monoxide", [])
        
        # Find current hour index
        now = datetime.now()
        current_hour_str = now.strftime("%Y-%m-%dT%H:00")
        
        start_idx = 0
        for i, t in enumerate(times):
            if t >= current_hour_str:
                start_idx = i
                break
        
        # Get next 'hours' forecasts
        forecasts = []
        for i in range(start_idx, min(start_idx + hours, len(times))):
            pm25 = pm25_values[i] if i < len(pm25_values) and pm25_values[i] is not None else 0
            pm10 = pm10_values[i] if i < len(pm10_values) and pm10_values[i] is not None else 0
            
            # Calculate Indian AQI from PM2.5 and PM10
            aqi_pm25 = calculate_indian_aqi_pm25(pm25)
            aqi_pm10 = calculate_indian_aqi_pm10(pm10)
            aqi = max(aqi_pm25, aqi_pm10)  # Indian AQI is the max of sub-indices
            
            forecasts.append({
                "time": times[i],
                "aqi": aqi,
                "pm2_5": round(pm25, 2),
                "pm10": round(pm10, 2),
                "no2": round(no2_values[i], 2) if i < len(no2_values) and no2_values[i] else 0,
                "so2": round(so2_values[i], 2) if i < len(so2_values) and so2_values[i] else 0,
                "o3": round(o3_values[i], 2) if i < len(o3_values) and o3_values[i] else 0,
                "co": round(co_values[i], 2) if i < len(co_values) and co_values[i] else 0
            })
        
        return forecasts
    except Exception as e:
        print(f"Open-Meteo API error: {e}")
        return None

# ===================== WEATHER FEATURES (FOR ML MODEL) =====================
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

@cache.memoize(timeout=3600)
def fetch_weather_features(lat, lon, hours=24):
    """Fetch weather data for enhanced ML predictions"""
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", 
                      "precipitation", "cloud_cover", "surface_pressure"],
            "forecast_days": 2,
            "timezone": "auto"
        }
        
        res = requests.get(WEATHER_API_URL, params=params, timeout=15)
        data = res.json()
        
        if "hourly" not in data:
            return None
        
        hourly = data["hourly"]
        times = hourly.get("time", [])
        
        # Find current hour index
        now = datetime.now()
        current_hour_str = now.strftime("%Y-%m-%dT%H:00")
        
        start_idx = 0
        for i, t in enumerate(times):
            if t >= current_hour_str:
                start_idx = i
                break
        
        weather_data = []
        for i in range(start_idx, min(start_idx + hours, len(times))):
            weather_data.append({
                "time": times[i],
                "temperature": hourly.get("temperature_2m", [0])[i] if i < len(hourly.get("temperature_2m", [])) else 0,
                "humidity": hourly.get("relative_humidity_2m", [0])[i] if i < len(hourly.get("relative_humidity_2m", [])) else 0,
                "wind_speed": hourly.get("wind_speed_10m", [0])[i] if i < len(hourly.get("wind_speed_10m", [])) else 0,
                "precipitation": hourly.get("precipitation", [0])[i] if i < len(hourly.get("precipitation", [])) else 0,
                "cloud_cover": hourly.get("cloud_cover", [0])[i] if i < len(hourly.get("cloud_cover", [])) else 0,
                "pressure": hourly.get("surface_pressure", [1013])[i] if i < len(hourly.get("surface_pressure", [])) else 1013
            })
        
        return weather_data
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

def fetch_openmeteo_current(lat, lon):
    """Get current data from Open-Meteo as fallback"""
    try:
        forecasts = fetch_openmeteo_forecast(lat, lon, hours=1)
        if forecasts and len(forecasts) > 0:
            current = forecasts[0]
            return {
                "aqi_value": current["aqi"],
                "pm2_5": current["pm2_5"],
                "pm10": current["pm10"],
                "no2": current["no2"],
                "so2": current["so2"],
                "co": current["co"],
                "o3": current["o3"],
                "source": "open-meteo"
            }
        return None
    except:
        return None

# ===================== KALMAN FILTER FOR DATA FUSION =====================
class KalmanFilter:
    """
    Simple 1D Kalman Filter for AQI fusion.
    State: [estimated_aqi]
    """
    def __init__(self, initial_estimate=100, initial_error=50, process_noise=5, measurement_noise=10):
        self.estimate = initial_estimate
        self.error = initial_error  # Estimation error covariance
        self.Q = process_noise      # Process noise (model uncertainty)
        self.R = measurement_noise  # Measurement noise (sensor uncertainty)
    
    def predict(self, control_input=0):
        """Prediction step - estimate next state"""
        # For AQI, we assume random walk: x_k = x_{k-1} + noise
        # No control input in this simple model
        self.error = self.error + self.Q
        return self.estimate
    
    def update(self, measurement, measurement_noise=None):
        """Update step - incorporate new measurement"""
        if measurement_noise is not None:
            self.R = measurement_noise
        
        # Kalman Gain: how much to trust the measurement vs prediction
        K = self.error / (self.error + self.R)
        
        # Update estimate
        self.estimate = self.estimate + K * (measurement - self.estimate)
        
        # Update error covariance
        self.error = (1 - K) * self.error
        
        return self.estimate, K  # Return gain for debugging/logging

# ===================== MULTI-SOURCE DATA FUSION (KALMAN) =====================
def fuse_predictions(aqicn_current, openmeteo_forecasts, lstm_predictions=None):
    """
    Novelty 1: Multi-Source Data Fusion with Kalman Filtering
    Dynamically weighs sources based on their reliability and past performance.
    Returns predictions with calibrated confidence intervals.
    """
    if not openmeteo_forecasts:
        return None
    
    fused_predictions = []
    current_aqi = aqicn_current.get("aqi_value", 100) if aqicn_current else 100
    
    # Initialize Kalman Filter with AQICN current value (most trusted for t=0)
    kf = KalmanFilter(
        initial_estimate=current_aqi,
        initial_error=10,      # Low initial error since AQICN is reliable
        process_noise=3,       # AQI changes gradually hour-to-hour
        measurement_noise=15   # Default measurement noise
    )
    
    # Source reliability weights (used as inverse measurement noise)
    # Lower noise = higher trust
    SOURCE_NOISE = {
        "aqicn": 5,        # Very reliable for current data
        "openmeteo": 15,   # Good for forecasts
        "lstm": 20         # Model predictions have higher uncertainty
    }
    
    for i, forecast in enumerate(openmeteo_forecasts):
        # Step 1: Kalman Predict (time update)
        kf.predict()
        
        # Step 2: Fuse measurements from available sources
        measurements = []
        
        # Open-Meteo forecast (always available)
        openmeteo_aqi = forecast["aqi"]
        # Increase Open-Meteo noise with forecast horizon
        om_noise = SOURCE_NOISE["openmeteo"] + (i * 2)
        measurements.append((openmeteo_aqi, om_noise))
        
        # LSTM prediction (if available)
        if lstm_predictions and i < len(lstm_predictions):
            lstm_aqi = lstm_predictions[i].get("aqi", openmeteo_aqi)
            # LSTM noise increases with horizon but less steeply
            lstm_noise = SOURCE_NOISE["lstm"] + (i * 1.5)
            measurements.append((lstm_aqi, lstm_noise))
        
        # For t=0, also use AQICN current value with very low noise
        if i == 0 and aqicn_current:
            measurements.append((current_aqi, SOURCE_NOISE["aqicn"]))
        
        # Step 3: Sequential Kalman updates with all measurements
        total_gain = 0
        for measurement, noise in measurements:
            _, gain = kf.update(measurement, measurement_noise=noise)
            total_gain += gain
        
        fused_aqi = kf.estimate
        
        # Step 4: Calculate calibrated uncertainty
        # Base uncertainty from Kalman error covariance + horizon penalty
        base_uncertainty = max(5, kf.error)
        horizon_penalty = i * 1.5  # Smaller than before due to Kalman's inherent uncertainty tracking
        total_uncertainty = min(base_uncertainty + horizon_penalty, 50)
        
        future_time = datetime.now() + timedelta(hours=i+1)
        time_display = future_time.strftime("%H:00")
        
        fused_predictions.append({
            "hour": time_display,
            "aqi": int(fused_aqi),
            "aqi_lower": max(0, int(fused_aqi - total_uncertainty)),
            "aqi_upper": min(500, int(fused_aqi + total_uncertainty)),
            "uncertainty": round(total_uncertainty, 1),
            "kalman_gain": round(total_gain, 3),  # For debugging/analytics
            "status": get_aqi_status(int(fused_aqi)),
            "color": get_aqi_color(int(fused_aqi)),
            "pm2_5": forecast.get("pm2_5", 0),
            "pm10": forecast.get("pm10", 0)
        })
    
    return fused_predictions

# ===================== EXPLAINABLE AI (XAI) =====================
def calculate_feature_importance(current_data):
    """
    Novelty 2: Explainable AI
    Calculate approximate feature importance for AQI prediction
    Uses pollutant contributions to overall AQI
    """
    if not current_data:
        return {}
    
    # Calculate individual AQI contributions
    pm25 = current_data.get("pm2_5", 0)
    pm10 = current_data.get("pm10", 0)
    no2 = current_data.get("no2", 0)
    so2 = current_data.get("so2", 0)
    o3 = current_data.get("o3", 0)
    co = current_data.get("co", 0)
    
    aqi_pm25 = calculate_indian_aqi_pm25(pm25)
    aqi_pm10 = calculate_indian_aqi_pm10(pm10)
    
    # Simplified AQI calculations for other pollutants
    aqi_no2 = min(int(no2 * 0.5), 500)
    aqi_so2 = min(int(so2 * 0.5), 500)
    aqi_o3 = min(int(o3 * 0.5), 500)
    aqi_co = min(int(co / 100), 500)
    
    total = aqi_pm25 + aqi_pm10 + aqi_no2 + aqi_so2 + aqi_o3 + aqi_co + 1  # +1 to avoid div by zero
    
    importance = {
        "PM2.5": {
            "value": pm25,
            "sub_aqi": aqi_pm25,
            "contribution": round((aqi_pm25 / total) * 100, 1),
            "impact": "high" if aqi_pm25 > 100 else "medium" if aqi_pm25 > 50 else "low"
        },
        "PM10": {
            "value": pm10,
            "sub_aqi": aqi_pm10,
            "contribution": round((aqi_pm10 / total) * 100, 1),
            "impact": "high" if aqi_pm10 > 100 else "medium" if aqi_pm10 > 50 else "low"
        },
        "NO2": {
            "value": no2,
            "sub_aqi": aqi_no2,
            "contribution": round((aqi_no2 / total) * 100, 1),
            "impact": "high" if aqi_no2 > 100 else "medium" if aqi_no2 > 50 else "low"
        },
        "SO2": {
            "value": so2,
            "sub_aqi": aqi_so2,
            "contribution": round((aqi_so2 / total) * 100, 1),
            "impact": "high" if aqi_so2 > 100 else "medium" if aqi_so2 > 50 else "low"
        },
        "O3": {
            "value": o3,
            "sub_aqi": aqi_o3,
            "contribution": round((aqi_o3 / total) * 100, 1),
            "impact": "high" if aqi_o3 > 100 else "medium" if aqi_o3 > 50 else "low"
        },
        "CO": {
            "value": co,
            "sub_aqi": aqi_co,
            "contribution": round((aqi_co / total) * 100, 1),
            "impact": "high" if aqi_co > 100 else "medium" if aqi_co > 50 else "low"
        }
    }
    
    # Sort by contribution
    sorted_importance = dict(sorted(importance.items(), 
                                    key=lambda x: x[1]["contribution"], 
                                    reverse=True))
    
    return sorted_importance

# ===================== ANOMALY DETECTION =====================
def detect_anomaly(current_aqi, city, historical_aqis=None):
    """
    Novelty 3: Anomaly Detection with Root Cause Analysis
    Uses IsolationForest to detect unusual AQI spikes
    """
    if historical_aqis is None:
        # Use typical AQI range for the city as baseline
        historical_aqis = [80, 90, 85, 95, 100, 88, 92, 87, 93, 91,
                          85, 88, 90, 95, 92, 87, 89, 91, 86, 94]
    
    if len(historical_aqis) < 10:
        return {"is_anomaly": False, "message": "Insufficient data"}
    
    # Prepare data for IsolationForest
    X = np.array(historical_aqis + [current_aqi]).reshape(-1, 1)
    
    # Fit IsolationForest
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X[:-1])  # Train on historical data
    
    # Predict on current value
    prediction = clf.predict([[current_aqi]])[0]
    score = clf.decision_function([[current_aqi]])[0]
    
    is_anomaly = bool(prediction == -1)
    
    result = {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(float(score), 3),
        "current_aqi": current_aqi,
        "historical_avg": round(np.mean(historical_aqis), 1),
        "deviation": round(current_aqi - np.mean(historical_aqis), 1)
    }
    
    # Root cause analysis
    if is_anomaly:
        hour = datetime.now().hour
        
        # Determine likely root cause based on time and deviation
        if 7 <= hour <= 10 or 17 <= hour <= 20:
            root_cause = "Rush hour traffic congestion"
        elif 10 <= hour <= 16:
            if current_aqi > 200:
                root_cause = "Industrial emissions or construction activity"
            else:
                root_cause = "Daytime urban activity"
        elif 21 <= hour or hour <= 5:
            root_cause = "Possible industrial night operations or weather inversion"
        else:
            root_cause = "Unknown - requires investigation"
        
        result["root_cause"] = root_cause
        result["severity"] = "high" if current_aqi > 300 else "medium" if current_aqi > 200 else "low"
        
        # Log anomaly to database
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''INSERT INTO anomaly_logs (city, aqi_value, anomaly_type, root_cause)
                        VALUES (?, ?, ?, ?)''', 
                     (city, current_aqi, result["severity"], root_cause))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging anomaly: {e}")
    
    return result

# ===================== LSTM PREDICTION WITH UNCERTAINTY =====================
def predict_with_lstm_uncertainty(current_aqi, lat=None, lon=None, n_samples=10):
    """
    Enhanced LSTM prediction with Monte Carlo Dropout for uncertainty
    """
    # Lazy load model
    loaded_model, loaded_scaler, loaded_config = get_model()
    
    if loaded_model is None:
        return None
    
    try:
        lookback = loaded_config['lookback']
        n_features = loaded_config['n_features']
        
        # Create synthetic sequence based on current AQI
        base_variation = np.random.normal(0, 3, lookback)
        sequence = np.array([max(10, min(500, current_aqi + v)) for v in base_variation])
        
        # Prepare input
        if n_features == 1:
            input_sequence = sequence.reshape(-1, 1)
        else:
            input_sequence = np.zeros((lookback, n_features))
            input_sequence[:, 0] = sequence
            current_hour = datetime.now().hour
            for i in range(lookback):
                hour = (current_hour - lookback + i) % 24
                if n_features > 1:
                    input_sequence[i, 1] = np.sin(2 * np.pi * hour / 24)
                if n_features > 2:
                    input_sequence[i, 2] = np.cos(2 * np.pi * hour / 24)
        
        # Scale input
        scaled_input = loaded_scaler.transform(input_sequence)
        scaled_input = scaled_input.reshape(1, lookback, n_features)
        
        # Monte Carlo Dropout - multiple forward passes
        predictions_samples = []
        for _ in range(n_samples):
            # Note: In production, use model(scaled_input, training=True) for MC Dropout
            prediction = loaded_model.predict(scaled_input, verbose=0)
            predictions_samples.append(prediction[0])
        
        # Calculate mean and uncertainty
        predictions_array = np.array(predictions_samples)
        mean_predictions = np.mean(predictions_array, axis=0)
        std_predictions = np.std(predictions_array, axis=0)
        
        # Inverse transform
        forecast_horizon = loaded_config.get('forecast_horizon', 24)
        dummy = np.zeros((1, forecast_horizon, n_features))
        dummy[0, :, 0] = mean_predictions
        prediction_actual = loaded_scaler.inverse_transform(dummy.reshape(-1, n_features))[:, 0]
        
        # Build predictions with uncertainty
        predictions = []
        for i, aqi in enumerate(prediction_actual):
            aqi = max(10, min(500, aqi))
            uncertainty = std_predictions[i] * 50  # Scale uncertainty
            
            predictions.append({
                "hour": f"+{i+1}h",
                "aqi": int(aqi),
                "aqi_lower": max(0, int(aqi - uncertainty)),
                "aqi_upper": min(500, int(aqi + uncertainty)),
                "uncertainty": round(uncertainty, 1),
                "status": get_aqi_status(int(aqi)),
                "color": get_aqi_color(int(aqi))
            })
        
        return predictions
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return None

# ===================== SCENARIO PREDICTIONS =====================
def predict_with_scenario(base_predictions, scenario):
    """
    What-if scenario predictions
    Scenarios: normal, high_traffic, industrial, weather_event
    """
    if not base_predictions:
        return None
    
    scenario_factors = {
        "normal": 1.0,
        "high_traffic": 1.3,  # 30% increase
        "industrial": 1.5,   # 50% increase
        "weather_event": 0.7,  # 30% decrease (rain washes pollutants)
        "diwali": 2.0,  # Festival pollution
        "lockdown": 0.5  # Reduced activity
    }
    
    factor = scenario_factors.get(scenario, 1.0)
    
    adjusted = []
    for pred in base_predictions:
        new_aqi = int(pred["aqi"] * factor)
        new_aqi = max(10, min(500, new_aqi))
        
        adjusted.append({
            **pred,
            "aqi": new_aqi,
            "aqi_lower": max(0, int(pred.get("aqi_lower", new_aqi - 10) * factor)),
            "aqi_upper": min(500, int(pred.get("aqi_upper", new_aqi + 10) * factor)),
            "status": get_aqi_status(new_aqi),
            "color": get_aqi_color(new_aqi),
            "scenario": scenario
        })
    
    return adjusted

# ===================== API ROUTES =====================

@app.route("/api/current")
def current():
    """Get current AQI using AQICN (primary) with Open-Meteo fallback"""
    try:
        city = request.args.get("city", "Bangalore")
        lat = request.args.get("lat")
        lon = request.args.get("lon")
        
        aqicn_data = None
        
        # 1. If lat/lon provided, use them directly
        if lat and lon:
            aqicn_data = fetch_aqicn_by_coords(lat, lon)
            if not aqicn_data:
                aqicn_data = fetch_openmeteo_current(float(lat), float(lon))
                if aqicn_data: 
                    aqicn_data["source"] = "open-meteo"
            
            # If we used coordinates, we might want to resolve the city name for display
            # But strictly speaking, the frontend just needs data. 
            # We can default city to "Current Location" if not provided or resolved elsewhere.
            if not city or city == "Bangalore":
                city = "Current Location"

        # 2. If no data yet (or no coords), try by City Name
        if not aqicn_data:
            # Get current data from AQICN first
            aqicn_data = fetch_aqicn_current(city)
            
            # If AQICN fails, try by coordinates
            if not aqicn_data:
                coords = get_coordinates(city)
                aqicn_data = fetch_aqicn_by_coords(coords["lat"], coords["lon"])
            
            # Fallback to Open-Meteo if AQICN fails
            if not aqicn_data:
                coords = get_coordinates(city)
                aqicn_data = fetch_openmeteo_current(coords["lat"], coords["lon"])
        
        if not aqicn_data:
            return jsonify({"error": "Unable to fetch air quality data"}), 500
        
        aqi_value = aqicn_data["aqi_value"]
        health = get_health_recommendations(aqi_value)

        return jsonify({
            "city": city,
            "current": {
                "aqi_value": aqi_value,
                "aqi_status": get_aqi_status(aqi_value),
                "aqi_color": get_aqi_color(aqi_value),
                "pm2_5": aqicn_data.get("pm2_5", 0),
                "pm10": aqicn_data.get("pm10", 0),
                "no2": aqicn_data.get("no2", 0),
                "so2": aqicn_data.get("so2", 0),
                "co": aqicn_data.get("co", 0),
                "o3": aqicn_data.get("o3", 0),
                "time": datetime.now().strftime("%H:%M"),
                "source": aqicn_data.get("source", "unknown")
            },
            "activities": get_activity_recommendations(aqi_value),
            "health": health
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/predict")
def predict():
    """Get 24-hour AQI predictions using fused data sources"""
    try:
        city = request.args.get("city", "Bangalore")
        hours = int(request.args.get("hours", 24))
        scenario = request.args.get("scenario", "normal")
        
        # Get coordinates
        coords = get_coordinates(city)
        
        # Get current data from AQICN
        aqicn_current = fetch_aqicn_current(city)
        if not aqicn_current:
            aqicn_current = fetch_aqicn_by_coords(coords["lat"], coords["lon"])
        
        # Get Open-Meteo forecasts
        openmeteo_forecasts = fetch_openmeteo_forecast(coords["lat"], coords["lon"], hours)
        
        # Get LSTM predictions if model is loaded
        current_aqi = aqicn_current["aqi_value"] if aqicn_current else 100
        lstm_predictions = predict_with_lstm_uncertainty(current_aqi, coords["lat"], coords["lon"])
        
        # Fuse predictions from multiple sources
        fused = fuse_predictions(aqicn_current, openmeteo_forecasts, lstm_predictions)
        
        # Apply scenario if not normal
        if scenario != "normal" and fused:
            fused = predict_with_scenario(fused, scenario)
        
        if not fused:
            # Fallback to simple prediction
            fused = []
            for i in range(hours):
                variation = np.random.normal(0, 5)
                aqi = max(10, min(500, current_aqi + variation + i * 0.5))
                future_time = datetime.now() + timedelta(hours=i+1)
                fused.append({
                    "hour": future_time.strftime("%H:00"),
                    "aqi": int(aqi),
                    "aqi_lower": max(0, int(aqi - 10)),
                    "aqi_upper": min(500, int(aqi + 10)),
                    "status": get_aqi_status(int(aqi)),
                    "color": get_aqi_color(int(aqi))
                })
        
        # Log predictions for later validation
        if fused:
            log_prediction(city, fused)
        
        return jsonify(fused)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/forecast")
def forecast():
    """Public API endpoint for forecasts"""
    return predict()

@app.route("/api/explain")
def explain():
    """XAI endpoint - explain AQI composition"""
    try:
        city = request.args.get("city", "Bangalore")
        
        # Get current data
        aqicn_data = fetch_aqicn_current(city)
        if not aqicn_data:
            coords = get_coordinates(city)
            aqicn_data = fetch_openmeteo_current(coords["lat"], coords["lon"])
        
        if not aqicn_data:
            return jsonify({"error": "Unable to fetch data"}), 500
        
        # Calculate feature importance
        importance = calculate_feature_importance(aqicn_data)
        
        return jsonify({
            "city": city,
            "current_aqi": aqicn_data["aqi_value"],
            "feature_importance": importance,
            "explanation": f"The current AQI of {aqicn_data['aqi_value']} is primarily driven by {list(importance.keys())[0]} which contributes {list(importance.values())[0]['contribution']}% to the overall index."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/weather")
def weather():
    """Get weather data for a location (useful for ML features)"""
    try:
        city = request.args.get("city", "Bangalore")
        hours = int(request.args.get("hours", 24))
        
        coords = get_coordinates(city)
        weather_data = fetch_weather_features(coords["lat"], coords["lon"], hours)
        
        if not weather_data:
            return jsonify({"error": "Unable to fetch weather data"}), 500
        
        return jsonify({
            "city": city,
            "coordinates": {"lat": coords["lat"], "lon": coords["lon"]},
            "hourly_weather": weather_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/search")
def search_cities():
    """Autocomplete search for cities using Open-Meteo Geocoding"""
    try:
        query = request.args.get("q", "")
        if len(query) < 2:
            return jsonify([])
        
        # Open-Meteo Geocoding API
        url = "https://geocoding-api.open-meteo.com/v1/search"
        res = requests.get(url, params={
            "name": query,
            "count": 5,
            "language": "en",
            "format": "json"
        }, timeout=5)
        
        data = res.json()
        suggestions = []
        
        if "results" in data:
            for result in data["results"]:
                name = result.get("name")
                country = result.get("country", "")
                admin1 = result.get("admin1", "")  # State/Region
                
                display = f"{name}, {country}"
                if admin1:
                    display = f"{name}, {admin1}, {country}"
                    
                suggestions.append({
                    "name": name,
                    "display_name": display,
                    "lat": result.get("latitude"),
                    "lon": result.get("longitude")
                })
        
        return jsonify(suggestions)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/anomaly")
def anomaly():
    """Anomaly detection endpoint"""
    try:
        city = request.args.get("city", "Bangalore")
        
        # Get current AQI
        aqicn_data = fetch_aqicn_current(city)
        if not aqicn_data:
            coords = get_coordinates(city)
            aqicn_data = fetch_openmeteo_current(coords["lat"], coords["lon"])
        
        if not aqicn_data:
            return jsonify({"error": "Unable to fetch data"}), 500
        
        current_aqi = aqicn_data["aqi_value"]
        
        # Detect anomaly
        result = detect_anomaly(current_aqi, city)
        result["city"] = city
        result["timestamp"] = datetime.now().isoformat()
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/historical")
def historical():
    """Get historical AQI data (extended to 30 days)"""
    try:
        city = request.args.get("city", "Bangalore")
        days = int(request.args.get("days", 7))
        days = min(days, 30)  # Cap at 30 days
        
        coords = get_coordinates(city)
        
        # Use Open-Meteo historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ["pm2_5", "pm10"],
            "timezone": "auto"
        }
        
        res = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", 
                          params=params, timeout=15)
        data = res.json()
        
        historical = []
        if "hourly" in data:
            times = data["hourly"].get("time", [])
            pm25_values = data["hourly"].get("pm2_5", [])
            pm10_values = data["hourly"].get("pm10", [])
            
            # Group by day
            daily_data = {}
            for i, t in enumerate(times):
                date = t.split("T")[0]
                if date not in daily_data:
                    daily_data[date] = {"pm25": [], "pm10": []}
                
                pm25 = pm25_values[i] if i < len(pm25_values) and pm25_values[i] else 0
                pm10 = pm10_values[i] if i < len(pm10_values) and pm10_values[i] else 0
                daily_data[date]["pm25"].append(pm25)
                daily_data[date]["pm10"].append(pm10)
            
            # Calculate daily averages
            for date, values in daily_data.items():
                avg_pm25 = np.mean(values["pm25"]) if values["pm25"] else 0
                avg_pm10 = np.mean(values["pm10"]) if values["pm10"] else 0
                aqi = max(calculate_indian_aqi_pm25(avg_pm25), calculate_indian_aqi_pm10(avg_pm10))
                
                historical.append({
                    "date": date,
                    "aqi": int(aqi),
                    "pm2_5": round(avg_pm25, 2),
                    "pm10": round(avg_pm10, 2),
                    "status": get_aqi_status(int(aqi))
                })
        
        # Sort by date
        historical.sort(key=lambda x: x["date"])
        
        return jsonify(historical)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/scenario")
def scenario():
    """What-if scenario predictions"""
    try:
        city = request.args.get("city", "Bangalore")
        scenario_name = request.args.get("scenario", "normal")
        
        # Get base predictions
        coords = get_coordinates(city)
        openmeteo_forecasts = fetch_openmeteo_forecast(coords["lat"], coords["lon"], 24)
        
        aqicn_current = fetch_aqicn_current(city)
        fused = fuse_predictions(aqicn_current, openmeteo_forecasts)
        
        # Apply scenario
        result = predict_with_scenario(fused, scenario_name)
        
        return jsonify({
            "city": city,
            "scenario": scenario_name,
            "predictions": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/user/profile", methods=["GET", "POST"])
def user_profile():
    """User health profile management"""
    try:
        if request.method == "POST":
            data = request.get_json()
            email = data.get("email")
            name = data.get("name", "")
            health_conditions = data.get("health_conditions", "")
            alert_threshold = data.get("alert_threshold", 100)
            gdpr_consent = data.get("gdpr_consent", 0)
            
            if not email:
                return jsonify({"error": "Email is required"}), 400
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            # Upsert profile
            c.execute('''INSERT OR REPLACE INTO user_profiles 
                        (email, name, health_conditions, alert_threshold, gdpr_consent)
                        VALUES (?, ?, ?, ?, ?)''',
                     (email, name, health_conditions, alert_threshold, gdpr_consent))
            
            conn.commit()
            conn.close()
            
            return jsonify({"success": True, "message": "Profile saved"})
        
        else:  # GET
            email = request.args.get("email")
            if not email:
                return jsonify({"error": "Email parameter required"}), 400
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT * FROM user_profiles WHERE email = ?", (email,))
            row = c.fetchone()
            conn.close()
            
            if row:
                return jsonify({
                    "email": row[1],
                    "name": row[2],
                    "health_conditions": row[3],
                    "alert_threshold": row[4],
                    "gdpr_consent": row[6]
                })
            else:
                return jsonify({"error": "Profile not found"}), 404
                
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ===================== PREDICTION LOGGING & VALIDATION =====================
def log_prediction(city, predictions):
    """Log predictions to database for later validation"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        prediction_time = datetime.now().isoformat()
        
        for i, pred in enumerate(predictions[:24]):  # Log first 24 hours
            c.execute('''INSERT INTO prediction_logs 
                        (city, forecast_hour, predicted_aqi, prediction_time)
                        VALUES (?, ?, ?, ?)''',
                     (city, i+1, pred["aqi"], prediction_time))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging prediction: {e}")

def validate_predictions(city, hours_ago=24):
    """Compare past predictions with actual values"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get predictions made 'hours_ago' hours before
        cutoff_time = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
        
        c.execute('''SELECT id, forecast_hour, predicted_aqi, prediction_time 
                    FROM prediction_logs 
                    WHERE city = ? AND prediction_time < ? AND actual_aqi IS NULL
                    ORDER BY prediction_time DESC LIMIT 100''',
                 (city, cutoff_time))
        
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Error fetching predictions: {e}")
        return []

@app.route("/api/validate")
def validate():
    """Validation endpoint - compare predictions vs actuals"""
    try:
        city = request.args.get("city", "Bangalore")
        days = int(request.args.get("days", 7))
        
        # Get current actual AQI for validation
        aqicn_data = fetch_aqicn_current(city)
        actual_aqi = aqicn_data.get("aqi_value", 0) if aqicn_data else 0
        
        # Fetch unvalidated predictions from DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Update predictions that are now validatable (prediction_time + forecast_hour <= now)
        now = datetime.now()
        
        c.execute('''SELECT id, forecast_hour, predicted_aqi, prediction_time 
                    FROM prediction_logs 
                    WHERE city = ? AND actual_aqi IS NULL''', (city,))
        
        rows = c.fetchall()
        updated_count = 0
        errors = []
        
        for row in rows:
            pred_id, forecast_hour, predicted_aqi, pred_time_str = row
            try:
                pred_time = datetime.fromisoformat(pred_time_str)
                target_time = pred_time + timedelta(hours=forecast_hour)
                
                # Only validate if target time has passed
                if target_time <= now:
                    # Use current AQI as approximation (ideally fetch historical)
                    error = abs(predicted_aqi - actual_aqi)
                    
                    c.execute('''UPDATE prediction_logs 
                                SET actual_aqi = ?, validation_time = ?, error = ?
                                WHERE id = ?''',
                             (actual_aqi, now.isoformat(), error, pred_id))
                    
                    updated_count += 1
                    errors.append(error)
            except Exception as e:
                print(f"Validation error for id {pred_id}: {e}")
        
        conn.commit()
        
        # Calculate metrics from all validated predictions
        c.execute('''SELECT predicted_aqi, actual_aqi, error 
                    FROM prediction_logs 
                    WHERE city = ? AND error IS NOT NULL
                    ORDER BY validation_time DESC LIMIT ?''',
                 (city, days * 24))
        
        validated = c.fetchall()
        conn.close()
        
        if validated:
            errors_list = [row[2] for row in validated if row[2] is not None]
            mae = round(np.mean(errors_list), 2) if errors_list else 0
            rmse = round(np.sqrt(np.mean([e**2 for e in errors_list])), 2) if errors_list else 0
            
            return jsonify({
                "city": city,
                "current_actual_aqi": actual_aqi,
                "validated_predictions": len(validated),
                "newly_validated": updated_count,
                "metrics": {
                    "mae": mae,
                    "rmse": rmse,
                    "max_error": max(errors_list) if errors_list else 0,
                    "min_error": min(errors_list) if errors_list else 0
                },
                "recent_errors": errors_list[:10]  # Last 10 errors
            })
        else:
            return jsonify({
                "city": city,
                "message": "No validated predictions yet. Predictions will be validated as time passes.",
                "tip": "Call /api/predict first to log predictions, then wait for the forecast period to validate."
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "model_type": "improved_lstm" if MODEL_LOADED else "fallback",
        "apis": {
            "aqicn": "configured" if AQICN_TOKEN != "demo" else "demo_mode",
            "open_meteo": "active"
        },
        "features": [
            "multi_source_fusion",
            "explainable_ai",
            "anomaly_detection",
            "24h_forecast",
            "uncertainty_quantification"
        ],
        "timestamp": datetime.now().isoformat()
    })

# ===================== REAL-TIME WEBSOCKETS =====================
@socketio.on('connect')
def handle_connect():
    print('🔌 Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('🔌 Client disconnected')

@socketio.on('join')
def on_join(data):
    city = data.get('city')
    if city:
        join_room(city)
        print(f'👤 Client joined room: {city}')
        # Emit immediate update
        aqi_data = fetch_aqicn_current(city)
        if aqi_data:
            emit('aqi_update', aqi_data, to=city)

def background_polling():
    """Poll AQI data periodically and push updates"""
    print("🔄 Starting background polling service...")
    while True:
        try:
            # In a real app, we would track which cities have active subscribers
            # For demo, we just poll Bangalore + user's last searched cities
            cities_to_poll = ["Bangalore", "Delhi", "Mumbai"]
            
            for city in cities_to_poll:
                # Bypass cache to get fresh real-time check
                # (In production, use a more efficient strategy)
                data = fetch_aqicn_current(city)
                if data:
                    timestamp = datetime.now().isoformat()
                    # Add timestamp to data
                    data['push_timestamp'] = timestamp
                    socketio.emit('aqi_update', data, to=city)
            
            socketio.sleep(60)  # Poll every minute
        except Exception as e:
            print(f"Polling error: {e}")
            if socketio:
                socketio.sleep(60)
            else:
                time.sleep(60)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    
    if SOCKETIO_AVAILABLE and socketio:
        # Start background task
        socketio.start_background_task(background_polling)
        print("🚀 Starting Server with WebSockets...")
        socketio.run(app, debug=debug, host="0.0.0.0", port=port)
    else:
        print("🚀 Starting Server (no WebSockets)...")
        app.run(debug=debug, host="0.0.0.0", port=port)