import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_source_prioritization():
    print("\n--- Testing Source Prioritization ---")
    
    # 1. Indian City (Bangalore) -> Expect CPCB (Official)
    try:
        print("Fetching Bangalore (Indian)...")
        res = requests.get(f"{BASE_URL}/api/current?city=Bangalore")
        data = res.json()
        
        if res.status_code == 200:
            source = data['current']['source']
            print(f"✅ Bangalore Source: {source}")
            if "CPCB" in source or "Official" in source:
                print("   PASS: Correctly prioritized CPCB.")
            else:
                print("   WARN: Did not get CPCB (maybe API key/data issue or fallback).")
        else:
            print(f"❌ Failed to fetch Bangalore: {res.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

    # 2. International City (London) -> Expect AQICN (International)
    try:
        print("\nFetching London (International)...")
        res = requests.get(f"{BASE_URL}/api/current?city=London")
        data = res.json()
        
        if res.status_code == 200:
            source = data['current']['source']
            print(f"✅ London Source: {source}")
            if "International" in source or "AQICN" in source:
                print("   PASS: Correctly used AQICN.")
            else:
                print("   WARN: Unexpected source label.")
        else:
            print(f"❌ Failed to fetch London: {res.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")

def test_prediction_enhancements():
    print("\n--- Testing Prediction Enhancements ---")
    
    try:
        print("Fetching Predictions for Bangalore...")
        res = requests.get(f"{BASE_URL}/api/predict?city=Bangalore")
        data = res.json()
        
        if res.status_code == 200:
            preds = data.get("predictions", [])
            if not preds:
                print("❌ No predictions returned.")
                return

            first_pred = preds[0]
            print(f"✅ Received {len(preds)} hourly predictions.")
            
            # Check for new fields
            print("Checking for new fields in first prediction:")
            required_fields = ["confidence", "uncertainty", "kalman_gain"]
            missing = [f for f in required_fields if f not in first_pred]
            
            if not missing:
                print("   PASS: All new fields (confidence, uncertainty, kalman_gain) present.")
                print(f"   Sample Confidence: {first_pred['confidence']}%")
                print(f"   Sample Uncertainty: {first_pred['uncertainty']}")
            else:
                print(f"   FAIL: Missing fields: {missing}")
                
            # Check range
            aqi = first_pred['aqi']
            if 0 <= aqi <= 500:
                print(f"   PASS: AQI value {aqi} within valid range.")
            else:
                print(f"   FAIL: AQI value {aqi} out of range.")
                
        else:
             print(f"❌ Failed to fetch predictions: {res.status_code}")
             if "error" in data:
                 print(f"   Error: {data['error']}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Wait for server restart if needed...")
    time.sleep(2)
    test_source_prioritization()
    test_prediction_enhancements()
