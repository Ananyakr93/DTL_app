import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_typo_handling():
    print("\n--- Testing Typo Handling ---")
    
    test_cases = [
        {"input": "dehli", "expected": "New Delhi", "expected_source": "CPCB"},
        {"input": "bengaluru", "expected": "Bangalore", "expected_source": "CPCB"}, # Map to primary key
        # {"input": "calcutta", "expected": "Kolkata", "expected_source": "CPCB"}, # If alias works
    ]
    
    for case in test_cases:
        city_input = case["input"]
        print(f"\n🔍 Searching for '{city_input}'...")
        
        try:
            res = requests.get(f"{BASE_URL}/api/current?city={city_input}")
            
            if res.status_code == 200:
                data = res.json()
                resolved_city = data.get("city")
                source = data['current'].get('source', '')
                
                print(f"   ✅ Output City: '{resolved_city}'")
                print(f"   ✅ Source: '{source}'")
                
                # Validation
                if case["expected"] in resolved_city:
                    print(f"   PASS: Correctly resolved '{city_input}' to '{resolved_city}'.")
                else:
                    print(f"   WARN: Expected '{case['expected']}' but got '{resolved_city}'.")
                    
                if case["expected_source"] in source:
                    print(f"   PASS: Source is correct ({source}).")
                else:
                    print(f"   WARN: Source mismatch. Expected {case['expected_source']}.")
            else:
                print(f"   ❌ Failed to fetch data: {res.status_code}")
                
        except Exception as e:
            print(f"   ❌ Network error: {e}")

if __name__ == "__main__":
    time.sleep(2) # Wait for server
    test_typo_handling()
