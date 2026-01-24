
import requests
import os
import sys

# Ensure the current directory is in python path
sys.path.append(os.getcwd())

print("Testing CPCB API with timeout...")
try:
    CPCB_API_KEY = "579b464db66ec23bdd0000019cbe093c60694b174ef48b7e614a8098"
    CPCB_API_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
    
    city = "Bengaluru" 
    params = {
        'api-key': CPCB_API_KEY,
        'format': 'json',
        'limit': 10,
        'filters[city]': city
    }
    
    print(f"Fetching CPCB data for {city} (Timeout=10s)...")
    res = requests.get(CPCB_API_URL, params=params, timeout=10)
    data = res.json()
    print("Status Code:", res.status_code)
    
    if data.get('status') == 'ok':
        records = data.get('records', [])
        print(f"Found {len(records)} records.")
        
        pollutants = {'pm25': 0, 'pm10': 0}
        
        for rec in records:
            pollutant_id = rec.get('pollutant_id', '').lower().replace('.', '')
            val_str = rec.get('avg_value', rec.get('pollutant_avg', '0'))
            try:
                avg_value = float(val_str) if val_str not in ['NA', ''] else 0
            except:
                avg_value = 0
            
            print(f"  {rec.get('pollutant_id')}: {avg_value} (Raw: {val_str})")
            
            if 'pm25' in pollutant_id or pollutant_id == 'pm2.5':
                pollutants['pm25'] = max(pollutants['pm25'], avg_value)
            elif 'pm10' in pollutant_id:
                pollutants['pm10'] = max(pollutants['pm10'], avg_value)
                
        print("Final Parsed:", pollutants)
    else:
        print("API Error:", data)

except Exception as e:
    print(f"Error: {e}")
