# 🌿 AeroClean - Air Quality Prediction Dashboard

A **best-in-class, publish-worthy** real-time air quality prediction dashboard for Indian cities with **3 research novelties**.

![Version](https://img.shields.io/badge/version-3.0.0-green)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Key Features

- **24-Hour AQI Predictions** with uncertainty quantification
- **Multi-Source Data Fusion** (AQICN + Open-Meteo + LSTM)
- **Explainable AI (XAI)** for pollutant impact analysis
- **Anomaly Detection** with root cause identification
- **Interactive Maps** using Leaflet.js
- **PDF/Excel Report Generation**
- **Health Profiles** for personalized recommendations
- **Dark Mode** support

## 🔬 Research Novelties

| Novelty | Description | Publication Angle |
|---------|-------------|-------------------|
| **Multi-Source Data Fusion** | Bayesian ensemble of AQICN, Open-Meteo, and LSTM predictions with confidence intervals | Uncertainty-aware AQI forecasting |
| **Explainable AI** | Gradient-based feature importance showing pollutant contribution percentages | Interpretable deep learning for air quality |
| **Anomaly Detection** | IsolationForest with temporal pattern correlation for spike detection | Automated pollution event attribution |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone or navigate to the project
cd d:\DTL

# Install dependencies
pip install -r requirements.txt

# (Optional) Train the LSTM model
python train_lstm.py

# Start the server
python app.py
```

### Access the Dashboard
Open your browser and navigate to: `http://127.0.0.1:5000`

## 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/current?city=Bangalore` | Current AQI (AQICN) |
| `GET /api/predict?city=Bangalore&hours=24` | 24-hour predictions |
| `GET /api/explain?city=Bangalore` | XAI pollutant breakdown |
| `GET /api/anomaly?city=Bangalore` | Anomaly detection |
| `GET /api/historical?city=Bangalore&days=30` | Historical data |
| `GET /api/scenario?city=Bangalore&scenario=high_traffic` | What-if scenarios |
| `GET /api/health` | API health check |

## 🔑 API Configuration

### AQICN Token
Register at [aqicn.org/api](https://aqicn.org/api/) and set the token:

```bash
set AQICN_TOKEN=your_token_here  # Windows
export AQICN_TOKEN=your_token_here  # Linux/Mac
```

### Open-Meteo
No API key required (free tier).

## 📁 Project Structure

```
DTL/
├── app.py              # Flask backend with all APIs
├── train_lstm.py       # LSTM model training (24-hour forecast)
├── index.html          # Main dashboard
├── analytics.html      # Analytics with maps & XAI
├── devices.html        # Anomaly detection
├── reports.html        # PDF/Excel generation
├── settings.html       # User preferences
├── script.js           # Frontend logic
├── style.css           # Styling with dark mode
├── aqi_data.csv        # Training data
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## 🎯 Supported Indian Cities

Bangalore, Delhi, Mumbai, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad, Jaipur, Lucknow, and more.

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Forecast Horizon | 24 hours |
| Data Sources | 2 (AQICN + Open-Meteo) |
| Historical Data | 30 days |
| AQI Standard | Indian CPCB |

## 🐳 Docker Deployment

```bash
docker build -t aeroclean .
docker run -p 5000:5000 -e AQICN_TOKEN=your_token aeroclean
```

## ☁️ Render.com Deployment

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Manual Setup

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repository
4. Render auto-detects `render.yaml` configuration
5. Add environment variable:
   - `AQICN_TOKEN` = your token from [aqicn.org](https://aqicn.org/api/)
6. Deploy!

> **Render supports:** TensorFlow, WebSockets, persistent storage, and no cold starts.

## 📜 License

MIT License - see LICENSE file for details.

---

Built with ❤️ for cleaner air in India
