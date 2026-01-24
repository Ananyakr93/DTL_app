from app import match_city_name, is_indian_city, calculate_indian_aqi_pm25
import difflib

# Mock the API fetch logic to see what would happen
def simulate_logic(user_input):
    print(f"\n--- Processing Input: '{user_input}' ---")
    
    # 1. City Name Matching
    matched_name = match_city_name(user_input)
    print(f"1. Match Result: '{matched_name}'")
    
    final_city = matched_name if matched_name else user_input
    
    # 2. Country/Source Determination
    is_india = is_indian_city(final_city)
    print(f"2. Is Indian City? {is_india}")
    
    # 3. API Routing (Simulated from get_best_current_aqi)
    if is_india:
        print("3. Routing: PRIORITY -> CPCB API (Official)")
        
        # 4. CPCB Mapping (Simulated from fetch_cpcb_data)
        city_mapping = {
            'bangalore': 'Bengaluru',
            'bengaluru': 'Bengaluru',
            'delhi': 'Delhi',
            'new delhi': 'Delhi',
            'mumbai': 'Mumbai',
            'chennai': 'Chennai',
            'kolkata': 'Kolkata',
            'hyderabad': 'Hyderabad',
        }
        search_term = city_mapping.get(final_city.lower(), final_city.title())
        print(f"4. CPCB Search Term: '{search_term}'")
    else:
        print("3. Routing: PRIORITY -> AQICN API (International)")

if __name__ == "__main__":
    print("Verifying Location & API Logic...")
    test_inputs = ["Bangalore", "Bengaluru", "New Delhi", "London", "New York", "Pune"]
    for i in test_inputs:
        simulate_logic(i)
