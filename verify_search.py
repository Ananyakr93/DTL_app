
import sys
import os

# Adjust path to import app
sys.path.append('d:\\DTL')

from app import app, MONITORED_LOCATIONS

def verify_search():
    print("Verifying Search Endpoint...")
    client = app.test_client()
    
    # Test 1: Search for a major city (Bangalore)
    res = client.get('/api/search?q=Bangalore')
    data = res.get_json()
    print(f"Search 'Bangalore': Found {len(data)} results")
    if len(data) > 0 and data[0]['source'] == 'local':
        print("✅ Local cache working for Bangalore")
    else:
        print("❌ Local cache FAILED for Bangalore")
        print(data)

    # Test 2: Search for a sub-area (Whitefield)
    res = client.get('/api/search?q=Whitefield')
    data = res.get_json()
    print(f"Search 'Whitefield': Found {len(data)} results")
    found_whitefield = any("Whitefield" in d['name'] for d in data)
    if found_whitefield:
        print("✅ Sub-area 'Whitefield' found")
    else:
        print("❌ Sub-area 'Whitefield' NOT found")

def verify_map_data():
    print("\nVerifying Map Data Endpoint...")
    client = app.test_client()
    res = client.get('/api/map-data')
    data = res.get_json()
    print(f"Map Data: Returned {len(data)} locations")
    
    if len(data) == len(MONITORED_LOCATIONS):
        print(f"✅ Map data count matches MONITORED_LOCATIONS ({len(data)})")
    else:
        print(f"❌ Map data count mismatch: {len(data)} vs {len(MONITORED_LOCATIONS)}")

if __name__ == "__main__":
    verify_search()
    verify_map_data()
