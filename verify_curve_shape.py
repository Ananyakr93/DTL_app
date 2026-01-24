
import requests
import json

def verify_curve():
    try:
        # Use a fictional city to force fallback logic first, then real
        # Force fallback by using a city name that likely lacks history in OpenMeteo or forcing timeout? 
        # Actually, let's just check the current output for Bangalore.
        
        url = "http://127.0.0.1:5000/api/predict?city=Bangalore"
        print(f"Fetching prediction from {url}...")
        
        res = requests.get(url)
        data = res.json()
        predictions = data.get("predictions", [])
        
        if not predictions:
            print("❌ No predictions found.")
            return
            
        print("\n📊 24-Hour Forecast Curve:")
        print(f"{'Hour':<5} | {'AQI':<5} | {'Visual':<20}")
        print("-" * 35)
        
        aqi_values = [p['aqi'] for p in predictions]
        min_aqi = min(aqi_values)
        max_aqi = max(aqi_values)
        range_aqi = max_aqi - min_aqi if max_aqi != min_aqi else 1
        
        for i, p in enumerate(predictions):
            aqi = p['aqi']
            # Simple bar graph
            bar_len = int(((aqi - min_aqi) / range_aqi) * 20)
            bar = "█" * bar_len
            print(f"+{i+1}h  | {aqi:<5} | {bar}")
            
        # Check variation
        if range_aqi < 5:
            print("\n⚠️ WARNING: Curve is very flat (Range < 5). Still might look fake.")
        else:
            print(f"\n✅ Curve shows variation (Range: {range_aqi}).")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_curve()
