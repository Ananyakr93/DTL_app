# AeroClean Project Report

## Overview
AeroClean is a comprehensive air quality monitoring and prediction platform designed to provide actionable health insights tailored to the Indian context. By leveraging official CPCB data and advanced machine learning models (GRU with Attention), the platform delivers accurate real-time AQI tracking and 24-hour forecasts.

## Recent Enhancements (February 2026)

### 1. Advanced Health Insights Banner
-   **Granular Categories**: Implemented precise WAQI-standard breakpoints (0-50, 51-100, 101-150, etc.) with category-specific health messages.
-   **Always-Visible UI**: The health banner is now a permanent fixture on the dashboard, ensuring critical health information is never hidden behind hover interactions.
-   **Personalized Alerts**: Users with specific health conditions (Asthma, Heart Disease) see prioritized alerts at the top of the banner.

### 2. Cigarette Equivalence Metric
-   **Impact Visualization**: To make pollution levels relatable, we added a "Cigarette Equivalence" metric based on Berkeley Earth research.
-   **Calculation**: `PM2.5 / 22 µg/m³ ≈ 1 cigarette/day`.
-   **Display**: Shows a subtle indicator (e.g., "≈ 4+ cigarettes/day") when PM2.5 is the dominant pollutant and AQI levels are unsafe.

### 3. Enhanced Health Risk Visualization
-   **Dynamic Icons**: Lungs, Heart, and Sensitive Group icons now dynamically change color and opacity based on specific AQI thresholds, providing immediate visual cues for health risks.
-   **Improved Trend Analysis**: Sparkline trends now clearly state "Air quality worsening", "Improving", or "Stable" to help users plan outdoor activities.

### 4. India-Centric Optimization
-   **Global Cleanup**: Removed non-relevant global data sources from the search and heatmap to ensure a strictly India-focused user experience.
-   **Search Optimization**: Search functionality now prioritizes and restricts results to Indian stations.

## Technical Performance
-   **Model**: Advanced GRU with Attention Mechanism.
-   **Accuracy**: MAE: 25.49, RMSE: 43.58.
-   **Data Sources**: Primary integration with CPCB (official) and fallback to AQICN.

## Future Roadmap
-   **Hyper-local Forecasting**: Expanding prediction capabilities to street-level granularity using IoT sensor networks.
-   **Mobile App**: Developing a React Native mobile application for on-the-go alerts.
