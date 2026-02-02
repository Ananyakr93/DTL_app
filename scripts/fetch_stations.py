
import os
import sqlite3
import requests
import json
import time

# Configuration
CPCB_API_KEY = "579b464db66ec23bdd0000019cbe093c60694b174ef48b7e614a8098"
CPCB_API_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'aeroclean.db')

def init_db():
    print(f"Checking DB at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stations (
        id TEXT PRIMARY KEY,
        name TEXT,
        city TEXT,
        state TEXT,
        lat REAL,
        lon REAL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def fetch_stations():
    print("Fetching stations from CPCB API...")
    stations = {}
    
    # Pagination loop
    offset = 0
    limit = 500
    total_fetched = 0
    
    while True:
        params = {
            'api-key': CPCB_API_KEY,
            'format': 'json',
            'limit': limit,
            'offset': offset
        }
        
        try:
            print(f"Fetching offset {offset}...")
            res = requests.get(CPCB_API_URL, params=params, timeout=30)
            data = res.json()
            
            if data.get('status') != 'ok':
                print(f"Error: {data.get('message')}")
                break
                
            records = data.get('records', [])
            if not records:
                break
            
            for rec in records:
                # Key validation
                city = rec.get('city')
                station = rec.get('station')
                state = rec.get('state')
                lat = rec.get('latitude')
                lon = rec.get('longitude')
                
                if not city or not station:
                    continue
                    
                # Creating a unique ID for the station
                station_id = f"{city}_{station}".replace(" ", "_").lower()
                
                if station_id not in stations:
                    stations[station_id] = {
                        "id": station_id,
                        "name": station,
                        "city": city,
                        "state": state,
                        "lat": lat,
                        "lon": lon
                    }
            
            total_fetched += len(records)
            if len(records) < limit: # End of data
                break
                
            offset += limit
            time.sleep(1) # Be nice to the API
            
        except Exception as e:
            print(f"Request failed: {e}")
            break
            
    print(f"Found {len(stations)} unique stations.")
    return list(stations.values())

def geocode_missing_stations(stations):
    print("Geocoding missing coordinates...")
    
    updated_count = 0
    
    for s in stations:
        # If lat/lon is missing or invalid (some APIs return 'NA' or 0)
        try:
            lat = float(s['lat']) if s['lat'] and s['lat'] != 'NA' else 0
            lon = float(s['lon']) if s['lon'] and s['lon'] != 'NA' else 0
        except:
            lat, lon = 0, 0
            
        if lat == 0 or lon == 0:
            # Fallback Geocoding via Open-Meteo
            query = f"{s['name']}, {s['city']}, India"
            url = "https://geocoding-api.open-meteo.com/v1/search"
            try:
                res = requests.get(url, params={"name": query, "count": 1}, timeout=2)
                geo_data = res.json()
                if geo_data.get('results'):
                    res_lat = geo_data['results'][0]['latitude']
                    res_lon = geo_data['results'][0]['longitude']
                    s['lat'] = res_lat
                    s['lon'] = res_lon
                    updated_count += 1
                    # print(f"Geocoded: {s['name']} -> {res_lat}, {res_lon}")
                else:
                    # Try just city
                    res = requests.get(url, params={"name": s['city'] + ", India", "count": 1}, timeout=2)
                    geo_data = res.json()
                    if geo_data.get('results'):
                        s['lat'] = geo_data['results'][0]['latitude']
                        s['lon'] = geo_data['results'][0]['longitude']
                        updated_count += 1
            except Exception as e:
                print(f"Geocoding failed for {query}: {e}")
            
            time.sleep(0.2) # Rate limit
    
    print(f"Geocoded {updated_count} stations.")
    return stations

def save_to_db(stations):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    count = 0
    for s in stations:
        try:
            # Ensure safe floats
            lat = float(s['lat']) if s['lat'] and s['lat'] != 'NA' else 0
            lon = float(s['lon']) if s['lon'] and s['lon'] != 'NA' else 0
            
            c.execute('''INSERT OR REPLACE INTO stations (id, name, city, state, lat, lon) 
                         VALUES (?, ?, ?, ?, ?, ?)''', 
                      (s['id'], s['name'], s['city'], s['state'], lat, lon))
            count += 1
        except Exception as e:
            print(f"Error saving {s['name']}: {e}")
            
    conn.commit()
    conn.close()
    print(f"Saved {count} stations to database.")

if __name__ == "__main__":
    init_db()
    stations = fetch_stations()
    if stations:
        # Check if we need geocoding (basic check on first few)
        # Using a subset for geocoding to save time if list is huge
        # But realistically, let's do all if they are missing
        stations = geocode_missing_stations(stations)
        save_to_db(stations)
    else:
        print("No stations found. Check API key or connectivity.")
