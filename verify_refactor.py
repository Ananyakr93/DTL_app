
import sys
import time
import json
import logging

# Setup logging to see app logs
logging.basicConfig(level=logging.INFO)

try:
    from app import app
    print("✅ App imported successfully")
except ImportError as e:
    print(f"❌ Failed to import app: {e}")
    sys.exit(1)

def verify_predict_latency():
    print("\n--- Verifying /api/predict Latency ---")
    client = app.test_client()
    
    # Warmup
    client.get('/api/predict?city=Bangalore')
    
    start_time = time.time()
    response = client.get('/api/predict?city=Bangalore&lat=12.9716&lon=77.5946')
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"⏱️ Response Time: {duration:.4f} seconds")
    
    if response.status_code == 200:
        print("✅ Status 200 OK")
        data = response.json
        print(f"📦 Payload keys: {list(data.keys())}")
        if 'aqi_value' in data or 'pm2_5' in data or isinstance(data, list):
             print("✅ Data structure looks valid")
        else:
             print("⚠️ Unexpected data structure")
             print(json.dumps(data, indent=2)[:500])
    else:
        print(f"❌ Failed with status {response.status_code}")
        print(response.data.decode())

def verify_current_fallback():
    print("\n--- Verifying /api/current Fallback Logic ---")
    # To test fallback, we rely on the implementation prioritizing specific sources.
    # This is hard to "prove" without mocking, but we can ensure it returns valid data.
    client = app.test_client()
    response = client.get('/api/current?city=Bangalore')
    
    if response.status_code == 200:
        data = response.json
        print(f"✅ /api/current returned: {data.get('source', 'unknown')} source")
    else:
        print(f"❌ /api/current failed: {response.status_code}")

if __name__ == "__main__":
    verify_predict_latency()
    verify_current_fallback()
