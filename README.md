# 🌬️ AeroClean Dashboard
### *Advanced Air Quality Monitoring & 24-Hour Forecast Platform*

AeroClean is a state-of-the-art air quality intelligence platform designed specifically for the Indian context. It combines official government data (CPCB) with advanced deep learning models (GRU + Attention) to provide real-time monitoring and highly accurate 24-hour AQI predictions.

---

## 🚀 Key Features

### 📊 Real-Time Monitoring
- **Dual-Source Data**: Intelligence-driven switching between **CPCB (Official Indian Data)** and **AQICN (WAQI Fallback)**.
- **Location-Aware**: Automatic detection of the nearest monitoring station with a 100km radius tolerance.
- **Live Pollutant Breakdown**: Detailed metrics for PM2.5, PM10, NO2, SO2, CO, and O3.

### 🔮 24-Hour Predictive Analytics
- **Deep Learning Forecasts**: Powered by an advanced **GRU (Gated Recurrent Unit)** model with an **Attention Mechanism**.
- **Scenario Simulation**: Test how air quality changes under "High Traffic", "Rain Event", or "Festival (Diwali)" scenarios.
- **Uncertainty Mapping**: visual indicator of prediction confidence intervals.

### 🏥 Hyper-Personalized Health Insights
- **Health Analogies**: Visualizes pollution impact through "Cigarette Equivalents" (Berkeley Earth Research).
- **Personalized Logic**: Alerts tailored for specific health conditions (Asthma, Heart Disease, Children, Elderly).
- **CPCB-Standard Guidance**: Automated activity recommendations (e.g., "Limit outdoor exercise") based on official Indian breakpoints.

### 🗺️ Interactive Visualizations
- **AQI Heatmap**: Leaflet-based map visualizing pollution levels across major Indian cities.
- **Analytics Dashboard**: Historical trend analysis with 'Improving' vs 'Worsening' trend detection.
- **Glassmorphism UI**: A premium, responsive interface with deep dark mode support and vibrant micro-animations.

---

## 🛠️ Tech Stack

- **Backend**: Python (Flask, Flask-SocketIO, Gunicorn, Gevent)
- **Frontend**: React 18, TypeScript, TailwindCSS, Vite
- **Machine Learning**: TensorFlow/Keras (GRU + Attention), Scikit-learn, Pandas, NumPy
- **Database**: SQLite (Profile management & Historical logging)
- **APIs**: CPCB (National Air Quality Index), AQICN, Open-Meteo (Geocoding)

---

## � Project Structure

```text
DTL/
├── src/                    # Flask Backend Application
│   ├── app.py              # Main Entry Point & API Logic
│   └── start_public.py     # Public tunnel starter
├── frontend/               # React Frontend (Vite + TS)
│   ├── src/components/     # UI Components (Map, Analytics, Modals)
│   └── src/utils.ts        # Common logic & CPCB formulas
├── models/                 # Pre-trained ML Models (.keras & .pkl)
├── data/                   # Historical datasets & SQLite DB
├── scripts/                # Utility scripts for data/cities
└── static/                 # Production-build Frontend Assets
```

---

## 🏁 Installation & Execution

### 1. Prerequisites
- **Python 3.10.x**
- **Node.js 18+**

### 2. Basic Setup (Local)
```bash
# Clone the repository and enter directory
cd DTL

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Integrated Production Mode
For the best experience, build the frontend and serve it via Flask:
```bash
# 1. Build Frontend
cd frontend
npm install
npm run build
cd ..

# 2. Sync Build Assets (Windows)
cp frontend/dist/index.html templates/index.html
cp -r frontend/dist/assets/* static/assets/

# 3. Start Backend
cd src
python app.py
```
App will be live at: `http://localhost:5000`

---

## ☁️ Deployment (Railway / Render)

This project is pre-configured for **Railway** deployment using the `railway.json` and a specific Production Entry Point.

1.  **Repository**: Connect your GitHub repo to Railway.
2.  **Environment Variables**:
    - `AQICN_TOKEN`: Your API token.
    - `CPCB_API_KEY`: Your official CPCB key.
    - `RAILWAY_ENVIRONMENT`: Set to `production`.
3.  **Start Command**: `gunicorn --worker-class gevent --workers 1 --bind 0.0.0.0:$PORT src.app:app`
4.  **Storage**: Create a Volume and mount it to `/data` for SQLite persistence.

---

## 📈 AQI Standard Coverage
AeroClean strictly follows the **Indian CPCB National AQI (NAQI)** standards:

| AQI Range | Category | Color | Impact |
|:---|:---|:---|:---|
| 0 - 50 | Good | 🟢 Green | Minimal Impact |
| 51 - 100 | Satisfactory | 🟢 Lime | Minor breathing discomfort to sensitive people |
| 101 - 200| Moderate | 🟡 Yellow | Breathing discomfort with lung/heart disease |
| 201 - 300| Poor | 🟠 Orange | Breathing discomfort on prolonged exposure |
| 301 - 400| Very Poor | 🔴 Red | Respiratory illness on prolonged exposure |
| 400+ | Severe | 🟣 Purple | Affects healthy people; serious impact on existing diseases |

---

## 📄 License & Attribution
- **Data Source**: CPCB (Central Pollution Control Board, India).
- **Models**: Built by the AeroClean Research Team using Berkeley Earth health analogies.
- **License**: Proprietary - All Rights Reserved.
