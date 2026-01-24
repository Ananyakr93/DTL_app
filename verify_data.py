
import sys
sys.path.append('d:\\DTL')
try:
    from indian_cities import INDIAN_CITIES
    print(f"Imported INDIAN_CITIES. Type: {type(INDIAN_CITIES)}")
    print(f"Length: {len(INDIAN_CITIES)}")
    
    # Check for any non-dict
    for i, item in enumerate(INDIAN_CITIES):
        if not isinstance(item, dict):
            print(f"❌ Item {i} is {type(item)}: {repr(item)}")
            # Print neighbors
            start = max(0, i-2)
            end = min(len(INDIAN_CITIES), i+3)
            print(f"Context: {INDIAN_CITIES[start:end]}")
            break
    else:
        print("✅ All items are dictionaries.")
        
    # Check Faridabad specifically
    print("\nChecking for 'Faridabad'...")
    found = False
    for item in INDIAN_CITIES:
        if isinstance(item, dict) and item.get('name') == 'Faridabad':
            print(f"Found Faridabad entry: {item}")
            found = True
    if not found:
        print("❌ Faridabad entry NOT found as dict.")
        
except Exception as e:
    print(f"❌ Import failed: {e}")
