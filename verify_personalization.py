
import unittest
import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

from app import get_personalized_risk

class TestPersonalizationLogic(unittest.TestCase):
    def test_normal_mode(self):
        # Good AQI
        r = get_personalized_risk(40, "normal")
        self.assertEqual(r["risk_level"], "Low")
        
        # Moderate AQI (150) -> Medium risk for normal
        r = get_personalized_risk(150, "normal")
        self.assertEqual(r["risk_level"], "Medium")
        
        # Poor AQI (250) -> High risk for normal
        r = get_personalized_risk(250, "normal")
        self.assertEqual(r["risk_level"], "High")
        self.assertIn("mask", str(r["tips"]).lower())

    def test_asthma_mode(self):
        # Moderate AQI (150) -> High risk for asthma
        r = get_personalized_risk(150, "asthma")
        self.assertEqual(r["risk_level"], "High")
        self.assertIn("inhaler", str(r["tips"]).lower())
        
        # Poor AQI (250) -> Severe risk? Or High?
        # Logic: 201-300: High if not sensitive else Severe?
        # Let's check impl: 
        # elif 201 <= aqi <= 300: risk["risk_level"] = "High" if not is_sensitive else "Severe"
        r = get_personalized_risk(250, "asthma")
        self.assertEqual(r["risk_level"], "Severe")

    def test_elderly_mode(self):
        r = get_personalized_risk(150, "elderly")
        self.assertEqual(r["risk_level"], "High")
        self.assertIn("hydration", str(r["tips"]).lower()) # check if hydration mentioned or similar?
        # Tips: ["Limit prolonged outdoor walks", "Stay hydrated", ...]
        self.assertTrue(any("hydra" in t.lower() for t in r["tips"]))

if __name__ == '__main__':
    unittest.main()
