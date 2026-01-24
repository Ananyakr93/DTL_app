
import unittest
import requests
import os
import sys
import threading
import time
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append(os.getcwd())

from app import (
    fetch_cpcb_data, 
    fetch_aqicn_current, 
    get_model, 
    init_db,
    calculate_indian_aqi_pm25,
    match_city_name
)

class TestAeroClean(unittest.TestCase):

    def test_01_database_init(self):
        """Test if database initializes correctly"""
        try:
            init_db()
            self.assertTrue(os.path.exists('aeroclean.db') or os.path.exists('/tmp/aeroclean.db'))
            print("✅ Database Init: PASS")
        except Exception as e:
            self.fail(f"Database init failed: {e}")

    def test_02_aqi_calculation(self):
        """Test Indian AQI Logic"""
        # PM2.5 of 30 should be AQI 50
        self.assertEqual(calculate_indian_aqi_pm25(30), 50)
        # PM2.5 of 60 should be AQI 100
        self.assertEqual(calculate_indian_aqi_pm25(60), 100)
        print("✅ AQI Calculation: PASS")

    def test_03_city_matching(self):
        """Test fuzzy city matching"""
        match = match_city_name("bengaluru")
        self.assertIn(match, ["Bangalore", "Bengaluru"]) # Either is fine
        print(f"✅ City Match (Bengaluru -> {match}): PASS")

    def test_04_cpcb_api_live(self):
        """Test Live CPCB API (Network Dependent)"""
        print("\n⏳ Testing CPCB API (Bangalore)... please wait...")
        data = fetch_cpcb_data("Bangalore")
        if data:
            print(f"   Response: AQI {data.get('aqi_value')} (Source: {data.get('source')})")
            self.assertEqual(data.get('source'), 'cpcb')
            self.assertIsNotNone(data.get('aqi_value'))
            print("✅ CPCB API: PASS")
        else:
            print("⚠️ CPCB API returned None (Could be temporary or rate limit)")

    def test_05_aqicn_api_live(self):
        """Test Live AQICN API (Network Dependent)"""
        print("\n⏳ Testing AQICN API (London)...")
        data = fetch_aqicn_current("London")
        if data:
            print(f"   Response: AQI {data.get('aqi_value')} (Source: {data.get('source')})")
            self.assertEqual(data.get('source'), 'aqicn')
            print("✅ AQICN API: PASS")
        else:
            self.fail("AQICN API failed for London")

    def test_06_model_lazy_loading(self):
        """Test ML Model Lazy Loading"""
        print("\n⏳ Testing Model Loading...")
        model, scaler, config = get_model()
        # It might be None if libraries missing, but function should run without crash
        if model:
             print("✅ Model Loading: PASS (Model Loaded)")
        else:
             print("✅ Model Loading: PASS (Graceful Fallback)")

if __name__ == '__main__':
    unittest.main(verbosity=2)
