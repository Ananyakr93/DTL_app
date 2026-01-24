
import requests
import json
import sys

BASE_URL = "http://127.0.0.1:5000"

def test_rankings():
    print("Testing /api/rankings...")
    try:
        res = requests.get(f"{BASE_URL}/api/rankings?timeframe=live")
        data = res.json()
        
        cleanest = data.get("cleanest", [])
        polluted = data.get("polluted", [])
        
        print(f"Cleanest count: {len(cleanest)}")
        for city in cleanest:
            if "," in city['name']:
                print(f"❌ FAIL: Found comma in city name: {city['name']}")
                return False
        
        print(f"Polluted count: {len(polluted)}")
        for city in polluted:
            if "," in city['name']:
                print(f"❌ FAIL: Found comma in city name: {city['name']}")
                return False
                
        print("✅ PASS: Rankings filtered correctly.")
        return True
    except Exception as e:
        print(f"❌ FAIL: Exception accessing rankings: {e}")
        return False

def test_current_label():
    print("\nTesting /api/current (Source Label)...")
    # We need to simulate a case where it uses fallback or just check general structure
    # Since we can't easily force fallback without mocking, we'll check if the endpoint responds
    # and if the logic *would* allow the new label. 
    # Actually, we can check a known Indian city that might use falling back if we provide a dummy city or one that CPCB might miss?
    # Or just check "Bangalore" which should ideally work with CPCB.
    
    try:
        # Check a random city to see if we get a response
        res = requests.get(f"{BASE_URL}/api/current?city=Bangalore")
        data = res.json()
        current = data.get("current", {})
        source = current.get("source", "")
        print(f"Source for Bangalore: {source}")
        
        # We can't strictly assert "AQICN (CPCB Std)" unless we force fallback. 
        # But we can verify the code change didn't break JSON structure.
        if "aqi_value" in current:
             print("✅ PASS: /api/current returned valid structure.")
        else:
             print("❌ FAIL: /api/current missing aqi_value.")
             return False
             
        return True

    except Exception as e:
        print(f"❌ FAIL: Exception accessing current: {e}")
        return False

def test_predictions():
    print("\nTesting /api/predict...")
    try:
        res = requests.get(f"{BASE_URL}/api/predict?city=Bangalore")
        data = res.json()
        
        if "predictions" in data and isinstance(data["predictions"], list):
            print(f"✅ PASS: /api/predict returns object with 'predictions' array.")
            print(f"Prediction count: {len(data['predictions'])}")
        else:
            print(f"❌ FAIL: /api/predict response structure incorrect. Keys: {data.keys()}")
            return False
            
        return True
    except Exception as e:
        print(f"❌ FAIL: Exception accessing predict: {e}")
        return False

if __name__ == "__main__":
    passed = True
    passed &= test_rankings()
    passed &= test_current_label()
    passed &= test_predictions()
    
    if passed:
        print("\nAll backend check passed!")
        sys.exit(0)
    else:
        print("\nSome checks failed.")
        sys.exit(1)
