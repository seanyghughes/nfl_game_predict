"""
Basic tests for NFL prediction project
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestBasicSetup(unittest.TestCase):
    """Test basic project setup and imports"""
    
    def test_imports(self):
        """Test that all modules can be imported"""
        try:
            from api.espn_client import ESPNClient
            from data.data_collector import NFLDataCollector
            from data.feature_engineering import NFLFeatureEngineer
            from ml.models import RandomForestModel, ModelTrainer
            from utils.logger import setup_logging
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_espn_client_creation(self):
        """Test ESPN client can be created"""
        try:
            from api.espn_client import ESPNClient
            client = ESPNClient()
            self.assertIsNotNone(client)
        except Exception as e:
            self.fail(f"ESPN client creation failed: {e}")
    
    def test_data_collector_creation(self):
        """Test data collector can be created"""
        try:
            from data.data_collector import NFLDataCollector
            collector = NFLDataCollector()
            self.assertIsNotNone(collector)
        except Exception as e:
            self.fail(f"Data collector creation failed: {e}")
    
    def test_feature_engineer_creation(self):
        """Test feature engineer can be created"""
        try:
            from data.feature_engineering import NFLFeatureEngineer
            engineer = NFLFeatureEngineer()
            self.assertIsNotNone(engineer)
        except Exception as e:
            self.fail(f"Feature engineer creation failed: {e}")
    
    def test_model_creation(self):
        """Test ML models can be created"""
        try:
            from ml.models import RandomForestModel, ModelTrainer
            model = RandomForestModel()
            trainer = ModelTrainer()
            self.assertIsNotNone(model)
            self.assertIsNotNone(trainer)
        except Exception as e:
            self.fail(f"Model creation failed: {e}")

if __name__ == '__main__':
    unittest.main() 