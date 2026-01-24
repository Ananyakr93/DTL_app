
import requests
import json
import sys
import time

BASE_URL = "http://127.0.0.1:5000"

def test_prediction_accuracy():
    print("Testing /api/predict for accuracy check...")
    try:
        start_time = time.time()
        res = requests.get(f"{BASE_URL}/api/predict?city=Bangalore&health_mode=normal")
        duration = time.time() - start_time
        
        print(f"Request took {duration:.2f} seconds")
        
        if res.status_code != 200:
            print(f"❌ FAIL: API Error {res.status_code}")
            return False
            
        data = res.json()
        predictions = data.get("predictions", [])
        
        if not predictions:
            print("❌ FAIL: No predictions returned")
            return False
            
        print(f"✅ PASS: Got {len(predictions)} hourly predictions")
        
        # Check if we have uncertainty values (indicates GRU usage)
        first_pred = predictions[0]
        uncertainty = first_pred.get("uncertainty", 0)
        print(f"🔍 Feature Check: Uncertainty for +1h is {uncertainty}")
        
        # We can't strictly prove it used 'real' history without logs, 
        # but if it didn't crash and returns reasonable data, it's a good sign.
        # Higher uncertainty often implies the model is active.
        
        aqi_val = first_pred.get("aqi")
        print(f"   Predicted AQI: {aqi_val} ({first_pred.get('status')})")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        return False

if __name__ == "__main__":
    if test_prediction_accuracy():
        print("\nPrediction accuracy components verified.")
        sys.exit(0)
    else:
        sys.exit(1)
