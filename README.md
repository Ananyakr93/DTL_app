# Air Quality Prediction Dashboard (AeroClean)

A comprehensive air quality monitoring and prediction platform using advanced machine learning models.

## 🚀 Features

- **Real-time AQI Monitoring**: Live air quality data from CPCB and AQICN APIs
- **24-Hour Predictions**: Advanced GRU model with attention mechanism (MAE: 25.49, RMSE: 43.58)
- **Health Insights Banner**: Always-visible, personalized health advice with granular AQI categories
- **Cigarette Equivalence**: Visualizes pollution impact in terms of cigarettes per day (Berkeley Earth)
- **Dynamic Risk Icons**: Interactive visualizations for lung and heart health risks
- **India-Centric Design**: Optimized search and heatmap strictly for Indian cities
- **Anomaly Detection**: Machine learning-based pollution spike detection

## 📁 Project Structure

```
DTL/
├── src/                    # Core application source code
│   ├── app.py              # Main Flask application
│   └── start_public.py     # Public server script
│
├── models/                 # ML model artifacts
│   ├── saved_models/       # Trained .keras/.h5 files
│   ├── compare_models.py   # Model evaluation script
│   ├── *.pkl               # Scalers and configs
│   └── *.png               # Training visualizations
│
├── scripts/                # Training and utility scripts
│   ├── train_advanced_gru.py
│   ├── train_gru.py
│   ├── train_lstm.py
│   └── indian_cities.py
│
├── templates/              # HTML templates (Jinja2)
│   ├── index.html          # Dashboard homepage
│   ├── analytics.html
│   ├── devices.html
│   ├── reports.html
│   └── settings.html
│
├── static/                 # Frontend assets
│   ├── css/style.css
│   ├── js/script.js
│   └── images/
│
├── data/                   # Data files (gitignored)
│   ├── aeroclean.db        # SQLite database
│   └── *.csv               # Training datasets
│
├── notebooks/              # Jupyter notebooks for exploration
│
├── docs/                   # Project documentation
│
├── venv/                   # Python virtual environment
├── requirements.txt        # Python dependencies
├── render.yaml             # Render.com deployment config
├── run.sh                  # Startup script

```

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DTL.git
cd DTL
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
cd src
python app.py
```

5. Open browser at: http://localhost:5000

## 🤖 Model Performance

| Model | MAE | RMSE | Status |
|-------|-----|------|--------|
| **Advanced GRU (Attention + STL)** | **25.49** | **43.58** | ✅ Target Met |
| GRU Baseline | 54.86 | 75.24 | - |
| LSTM Baseline | 56.18 | 76.12 | - |

## 📊 API Endpoints

- `GET /api/current?city=<city>` - Current AQI
- `GET /api/predict?city=<city>&hours=24` - 24-hour prediction
- `GET /api/historical?city=<city>&days=7` - Historical data
- `GET /api/health` - Server health check

## 🌐 Data Sources

- **Primary**: CPCB (Central Pollution Control Board) - Official Indian AQI
- **Fallback**: AQICN (World Air Quality Index) - Global coverage
- **Forecast**: Open-Meteo Air Quality API



## 👥 Authors

Design Thinking Lab Project - 2026
