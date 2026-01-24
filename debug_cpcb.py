import requests
import json
import os

# CPCB credentials from app.py
CPCB_API_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
CPCB_API_KEY = "579b464db66ec23bdd0000019cbe093c60694b174ef48b7e614a8098"

def debug_cpcb_bangalore():
    params = {
        'api-key': CPCB_API_KEY,
        'format': 'json',
        'limit': 50, # Get more records to see multiple stations
        'filters[city]': 'Bengaluru'
    }
    
    print(f"Fetching data for Bengaluru from {CPCB_API_URL}...")
    try:
        resp = requests.get(CPCB_API_URL, params=params, timeout=30)
        data = resp.json()
        
        if data.get('status') == 'ok':
            records = data.get('records', [])
            print(f"Found {len(records)} records.")
            
            # Group by station
            stations = {}
            for r in records:
                s_name = r.get('station')
                if s_name not in stations:
                    stations[s_name] = []
                stations[s_name].append(r)
                
            print("\nStation Analysis:")
            for s, recs in stations.items():
                print(f"--- Station: {s} ---")
                pm25 = 0
                pm10 = 0
                for r in recs:
                    pol = r.get('pollutant_id', '').lower()
                    val = r.get('avg_value', r.get('pollutant_avg', 0))
                    try:
                        val = float(val)
                    except:
                        val = 0
                    
                    print(f"  {r.get('pollutant_id')}: {val}")
                    
                    if 'pm2.5' in pol: pm25 = val
                    if 'pm10' in pol: pm10 = val
                
                # Manual AQI Calc roughly for checking
                print(f"  > PM2.5: {pm25}, PM10: {pm10}")
        else:
            print("API Error:", data)
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    debug_cpcb_bangalore()
