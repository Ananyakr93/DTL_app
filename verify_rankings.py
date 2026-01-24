
import time
import requests
import json

BASE_URL = "http://127.0.0.1:5000/api/rankings"

def test_rankings_latency():
    print("--- Testing Rankings API Latency ---")
    
    # 1. Test Live (Default)
    start = time.time()
    res = requests.get(BASE_URL)
    duration = time.time() - start
    print(f"Live Rankings: {duration:.4f}s | Status: {res.status_code}")
    
    if res.status_code != 200:
        print("❌ Live request failed")
        return

    # 2. Test Monthly (Async Fetch)
    print("\n--- Fetching Monthly Rankings (This triggers parallel API calls) ---")
    start = time.time()
    res = requests.get(f"{BASE_URL}?timeframe=monthly")
    duration = time.time() - start
    print(f"Monthly Rankings: {duration:.4f}s | Status: {res.status_code}")
    
    if res.status_code == 200:
        data = res.json()
        cleanest = data.get("cleanest", [])
        polluted = data.get("polluted", [])
        full_list = data.get("full_list", [])
        
        print(f"✅ Received {len(full_list)} cities in total.")
        print(f"✅ Top Cleanest: {[c['name'] for c in cleanest]}")
        print(f"✅ Top Polluted: {[c['name'] for c in polluted]}")
        
        if duration < 5.0: # Giving 5s buffer, aim is <3s
            print("✅ Performance is acceptable.")
        else:
            print("⚠️ Performance warning: > 5s")
            
    else:
        print(f"❌ Monthly request failed: {res.text}")

if __name__ == "__main__":
    try:
        test_rankings_latency()
    except Exception as e:
        print(f"Test failed: {e}")
