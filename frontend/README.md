# AeroClean Dashboard v2.0

A production-ready, real-time air quality monitoring dashboard built with React, TypeScript, and Tailwind CSS.

## ✨ Features

### 📊 Dashboard
- **Real-time AQI Display**: Shows current AQI with color-coded status (CPCB standard)
- **Pollutant Cards**: PM2.5, PM10, NO₂, SO₂, CO, O₃ with safe limits
- **24-Hour Predictions**: Interactive chart with confidence bands and scenario modeling
- **Health Recommendations**: Dynamic advice based on current air quality
- **Activity Suggestions**: What to do/avoid based on AQI levels

### 📈 Analytics Page
- **Historical Trends**: Line charts showing AQI over 7/30/90 days
- **Pollutant Analysis**: Individual pollutant trend lines
- **Contribution Pie Chart**: Average pollutant contribution breakdown
- **AQI Distribution**: Category distribution over time period
- **CSV Export**: Download historical data

### 📄 Reports Page
- **PDF Report Generation**: Printable reports with current data & forecasts
- **CSV Export**: Raw data export for analysis
- **Customizable**: Select date range and pollutants to include
- **Health Recommendations**: Include/exclude health advice

### ⚙️ Settings Page
- **Dark/Light Mode**: Toggle theme preference
- **Default Location**: Set your default city
- **Measurement Units**: Metric (µg/m³) or Imperial (ppm)
- **Alert Threshold**: Customize when to show health warnings
- **Refresh Interval**: 30s to 10 minutes

### 🔍 Search & Location
- **Autocomplete Search**: 80+ Indian cities with instant suggestions
- **Station Selection**: Choose specific monitoring stations in cities
- **Geolocation**: Detect your current location
- **No "Avg of Stations"**: Clean station names displayed

### 🌐 Data Sources
- **WAQI API**: Real-time AQI from World Air Quality Index
- **CPCB Standard**: Indian AQI categories and colors
- **Fallback**: Realistic mock data when API unavailable

## 🚀 Quick Start

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:3000
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── AQICard.tsx          # Main AQI display
│   │   ├── ActivitySection.tsx   # Activity recommendations
│   │   ├── AnalyticsPage.tsx    # Historical analytics
│   │   ├── Header.tsx           # Search + location
│   │   ├── HealthAlert.tsx      # Health warnings
│   │   ├── HealthSection.tsx    # Health advice
│   │   ├── LoadingState.tsx     # Loading skeleton
│   │   ├── PollutantCards.tsx   # Individual pollutants
│   │   ├── PredictionChart.tsx  # 24-hour forecast
│   │   ├── ReportsPage.tsx      # Report generation
│   │   ├── SettingsPage.tsx     # User preferences
│   │   └── Sidebar.tsx          # Navigation
│   ├── data/
│   │   └── cities.ts            # 80+ Indian cities & stations
│   ├── api.ts                   # WAQI API + fallback
│   ├── App.tsx                  # Main application
│   ├── index.css                # Tailwind + custom styles
│   ├── main.tsx                 # Entry point
│   ├── store.ts                 # Zustand state
│   ├── types.ts                 # TypeScript types
│   └── utils.ts                 # Helper functions
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## 🎨 AQI Color Scale (CPCB India)

| AQI Range | Category | Color |
|-----------|----------|-------|
| 0-50 | Good | 🟢 Green |
| 51-100 | Satisfactory | 🟢 Lime |
| 101-200 | Moderate | 🟡 Yellow |
| 201-300 | Poor | 🟠 Orange |
| 301-400 | Very Poor | 🔴 Red |
| 400+ | Severe | 🟣 Purple |

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_WAQI_TOKEN` | WAQI API token | `demo` |

Get your free API token at: https://aqicn.org/data-platform/token/

### Settings (Saved to localStorage)

- `isDarkMode`: Theme preference
- `defaultCity`: Startup location
- `units`: Measurement units
- `alertThreshold`: AQI level for warnings
- `enableNotifications`: Show health alerts
- `refreshInterval`: Auto-refresh interval

## 📱 Responsive Design

- **Desktop**: Full sidebar + multi-column layouts
- **Tablet**: Collapsed navigation + adaptive grids
- **Mobile**: Stack layouts + touch-friendly controls

## 🔄 Auto-Refresh

Data automatically refreshes based on settings (default: 60 seconds). A visible countdown shows time until next update.

## 🏙️ Supported Cities

Major cities with multiple stations:
- Delhi (10 stations)
- Mumbai (6 stations)
- Bangalore (6 stations)
- Chennai (3 stations)
- Kolkata (4 stations)
- Hyderabad (4 stations)

Plus 70+ more Indian cities including:
- Chikkamagaluru, Madikeri, Shimoga (Karnataka hill stations)
- Pune, Ahmedabad, Jaipur, Lucknow
- All state capitals

## 🛠️ Tech Stack

- **React 18** + **TypeScript**
- **Tailwind CSS** for styling
- **Zustand** for state management
- **Recharts** for data visualization
- **Lucide React** for icons
- **Vite** for development



---

Built with 💚 for cleaner air awareness
