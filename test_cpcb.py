
import sys
import os
import requests

# Ensure the current directory is in python path
sys.path.append(os.getcwd())

print("Testing CPCB API...")
try:
    CPCB_API_KEY = "579b464db66ec23bdd0000019cbe093c60694b174ef48b7e614a8098"
    CPCB_API_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
    
    city = "Bengaluru" # CPCB uses Bengaluru
    params = {
        'api-key': CPCB_API_KEY,
        'format': 'json',
        'limit': 10,
        'filters[city]': city
    }
    
    print(f"Fetching raw CPCB data for {city}...")
    res = requests.get(CPCB_API_URL, params=params)
    data = res.json()
    
    if data.get('status') == 'ok':
        records = data.get('records', [])
        print(f"Found {len(records)} records.")
        for r in records:
            print(r)
    else:
        print("API Error:", data)
except Exception as e:
    print(f"Error: {e}")
