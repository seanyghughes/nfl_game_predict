#!/usr/bin/env python3
"""
NFL Game Prediction Demo Script
Simple demonstration of the project functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def demo_feature_engineering():
    """Demonstrate feature engineering capabilities"""
    print("=== Feature Engineering Demo ===")
    
    try:
        from data.feature_engineering import NFLFeatureEngineer
        
        # Create sample data
        sample_game = {
            'id': '12345',
            'date': '2024-09-08T20:00:00Z',
            'season': {'type': 2},
            'week': {'number': 1},
            'competitions': [{
                'venue': {
                    'id': '123',
                    'capacity': 70000,
                    'indoor': False
                }
            }],
            'weather': {
                'temperature': 72,
                'windSpeed': 8,
                'humidity': 65
            }
        }
        
        sample_home_team = {
            'id': '1',
            'name': 'Buffalo Bills',
            'abbreviation': 'BUF',
            'record': [{'wins': 12, 'losses': 5, 'ties': 0, 'percentage': 0.706}],
            'rank': 3
        }
        
        sample_away_team = {
            'id': '2',
            'name': 'New York Jets',
            'abbreviation': 'NYJ',
            'record': [{'wins': 7, 'losses': 10, 'ties': 0, 'percentage': 0.412}],
            'rank': 18
        }
        
        # Initialize feature engineer
        engineer = NFLFeatureEngineer()
        
        # Extract features
        features = engineer.create_game_features(sample_game, sample_home_team, sample_away_team)
        
        print(f"Successfully extracted {len(features)} features!")
        print("\nSample features:")
        for key, value in list(features.items())[:10]:
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Feature engineering demo failed: {e}")

def demo_ml_models():
    """Demonstrate ML model capabilities"""
    print("\n=== Machine Learning Models Demo ===")
    
    try:
        from ml.models import RandomForestModel, ModelTrainer
        import pandas as pd
        import numpy as np
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X_sample = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create sample target (home team win/loss)
        y_sample = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
        
        print(f"Created sample dataset: {X_sample.shape}")
        
        # Initialize and train a model
        model = RandomForestModel(n_estimators=50, random_state=42)
        model.train(X_sample, y_sample)
        
        print("Model trained successfully!")
        
        # Make predictions
        predictions = model.predict(X_sample[:5])
        print(f"Sample predictions: {predictions}")
        
        # Test model trainer
        trainer = ModelTrainer(random_state=42)
        trainer.add_model(RandomForestModel(random_state=42))
        
        print("Model trainer initialized successfully!")
        
    except Exception as e:
        print(f"ML models demo failed: {e}")

def demo_espn_client():
    """Demonstrate ESPN client capabilities"""
    print("\n=== ESPN API Client Demo ===")
    
    try:
        from api.espn_client import ESPNClient
        
        # Initialize client
        client = ESPNClient()
        print("ESPN client initialized successfully!")
        
        # Note: We won't make actual API calls in the demo to avoid rate limiting
        print("Client is ready to make API calls to ESPN endpoints")
        print("Available methods:")
        print("  - get_teams()")
        print("  - get_scoreboard(year, season_type, week)")
        print("  - get_schedule(year, season_type, week)")
        print("  - get_game_details(event_id)")
        print("  - get_boxscore(event_id)")
        print("  - get_play_by_play(event_id)")
        
    except Exception as e:
        print(f"ESPN client demo failed: {e}")

def main():
    """Run all demos"""
    print("NFL Game Prediction Project Demo")
    print("=" * 40)
    
    demo_feature_engineering()
    demo_ml_models()
    demo_espn_client()
    
    print("\n" + "=" * 40)
    print("Demo completed successfully!")
    print("\nTo run the full pipeline:")
    print("  python main.py --mode full")
    print("\nTo collect data only:")
    print("  python main.py --mode collect")
    print("\nTo train models only:")
    print("  python main.py --mode train")

if __name__ == "__main__":
    main() 