# AeroClean Dashboard

A comprehensive air quality monitoring and prediction platform using advanced machine learning models (GRU + Attention) and a modern React frontend.

## 🚀 Features

- **Real-time AQI Monitoring**: Live air quality data from CPCB (India) and AQICN (Global) APIs.
- **24-Hour Predictions**: Advanced GRU model with attention mechanism (MAE: 25.49, RMSE: 43.58).
- **Health Insights Banner**: Always-visible, personalized health advice with granular AQI categories.
- **Cigarette Equivalence**: Visualizes pollution impact in terms of cigarettes per day (Berkeley Earth research).
- **Dynamic Risk Icons**: Interactive visualizations for lung and heart health risks.
- **India-Centric Design**: Optimized search and heatmap strictly for Indian cities.
- **Anomaly Detection**: Machine learning-based pollution spike detection.

## 📁 Project Structure

```
DTL/
├── src/                    # Flask Backend
│   └── app.py              # Main Application Entry
├── frontend/               # React Frontend (Vite)
├── models/                 # ML Models (.keras .pkl)
├── data/                   # Data Storage
└── requirements.txt        # Python Dependencies
```

## 🛠️ Execution Guide

Follow these steps to set up and run the project strictly.

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** (for frontend)

### Installation

#### 1. Backend Setup
Open a terminal in the root `DTL` directory:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Frontend Setup
Open a new terminal or navigate to the frontend folder:

```bash
cd frontend
npm install
```

---

### Running the Project

You can run the project in **Development Mode** (hot-reloading) or **Production Mode** (integrated).

#### Option A: Development Mode (Recommended for Editing)
Run backend and frontend in separate terminals.

**Terminal 1 (Backend - Flask):**
```bash
# Ensure venv is active
cd src
python app.py
```
*Backend runs on: http://127.0.0.1:5000*

**Terminal 2 (Frontend - React/Vite):**
```bash
cd frontend
npm run dev
```
*Frontend runs on: http://localhost:5173 (or port shown in terminal)*

#### Option B: Production Mode (Integrated)
Build the frontend and serve it purely via Flask.

**1. Build the Frontend:**
```bash
cd frontend
npm run build
```

**2. Deploy Artifacts (Windows PowerShell):**
Copy the built React files to Flask's template/static directories.

```powershell
# From DTL root directory:
Copy-Item frontend\dist\index.html -Destination templates\index.html -Force
Copy-Item -Recurse -Force frontend\dist\assets static\
```

**3. Run the Server:**
```bash
cd src
python app.py
```
Access the full app at: **http://127.0.0.1:5000**

## 📄 License
*Commercial / Proprietary - All Rights Reserved*
