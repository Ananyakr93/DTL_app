# Implementation Plan - Enhance CPCB Fetching & Dashboard Accuracy

This plan outlines the steps to improve the reliability and accuracy of the AQI dashboard, focusing on better CPCB data integration for Indian cities, robust fallbacks, and clearer frontend feedback.

## User Review Required

> [!IMPORTANT]
> I will be expanding the `indian_cities.py` list with more major Indian cities to improve coverage.
> I will also refine the `get_best_current_aqi` logic to prioritized CPCB data and fallback to AQICN more gracefully.

- **Gaps**: I am assuming the CPCB API key provided in `app.py` is valid and has sufficient quota. 
- **Geolocation**: I will rely on the existing `INDIAN_CITIES` coordinates for "nearest city" matching if precise CPCB station coordinates aren't available in the filtered API response.

## Proposed Changes

### 1. Source Code (`src/app.py`)

- **Refine `get_best_current_aqi`**:
    - Strictly enforce CPCB priority for Indian locations.
    - Implement a "station-level" check if city-level average is missing or dubious.
    - Ensure cache invalidation or shorter TTLs for real-time responsiveness.
- **Enhance `fetch_cpcb_data_async`**:
    - Improve error handling for "NA" or empty values which might be causing the "inflated values" (e.g. treating them as 0 then averaging might be wrong, or some parsing issue).
    - Add logic to handle cases where the API returns success but 0 records.
- **Expand `indian_cities.py`**:
    - Add approximately 20-30 more major Indian cities/towns to coverage.
    - Use `difflib` more effectively for fuzzy matching in `match_city_name`.

### 2. Frontend (`static/js/script.js`)

- **Update `updateUI`**:
    - Explicitly show the data source (CPCB vs AQICN).
    - Add specific styling/badges for the source.
- **Error Handling**:
    - If data values are null/undefined, display "Data Unavailable" or "N/A" instead of "--" or "undefined".

### 3. Verification

- **Manual Testing**:
    - Run the app and check `Bangalore`, `Delhi` (should hit CPCB).
    - Check an international city like `New York` (should hit AQICN).
    - Check a typo like `Bngalore` (should fuzzy match and hit CPCB).
