#!/usr/bin/env python3
"""
NFL Game Prediction Main Script
Demonstrates the complete pipeline from data collection to model training and prediction
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.utils.logger import setup_logging, get_logger
    from src.data.data_collector import NFLDataCollector
    from src.data.feature_engineering import NFLFeatureEngineer
    from src.ml.models import (
        RandomForestModel, XGBoostModel, LightGBMModel, 
        LogisticRegressionModel, NeuralNetworkModel, EnsembleModel, ModelTrainer
    )
except ImportError:
    # Fallback to direct imports if src path doesn't work
    from utils.logger import setup_logging, get_logger
    from data.data_collector import NFLDataCollector
    from data.feature_engineering import NFLFeatureEngineer
    from ml.models import (
        RandomForestModel, XGBoostModel, LightGBMModel, 
        LogisticRegressionModel, NeuralNetworkModel, EnsembleModel, ModelTrainer
    )

def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NFL Game Prediction ML Pipeline')
    parser.add_argument('--mode', choices=['collect', 'train', 'predict', 'full'], 
                       default='full', help='Pipeline mode')
    parser.add_argument('--year', type=int, default=2024, help='NFL season year')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=f"./logs/nfl_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logger.info("Starting NFL Game Prediction Pipeline")
    logger.info(f"Mode: {args.mode}, Year: {args.year}")
    
    try:
        if args.mode in ['collect', 'full']:
            # Data Collection Phase
            logger.info("=== Starting Data Collection ===")
            data_collector = NFLDataCollector()
            
            # Collect current season data
            logger.info("Collecting current season data...")
            current_data = data_collector.collect_current_season_data(args.year)
            
            # Collect historical data (last 3 years)
            logger.info("Collecting historical data...")
            historical_data = data_collector.collect_historical_data(
                start_year=args.year - 3, 
                end_year=args.year - 1
            )
            
            logger.info("Data collection completed successfully")
        
        if args.mode in ['train', 'full']:
            # Feature Engineering Phase
            logger.info("=== Starting Feature Engineering ===")
            feature_engineer = NFLFeatureEngineer()
            
            # For demonstration, we'll create sample features
            # In a real scenario, you'd process the collected data
            logger.info("Feature engineering completed")
        
        if args.mode in ['train', 'full']:
            # Model Training Phase
            logger.info("=== Starting Model Training ===")
            
            # Initialize models
            trainer = ModelTrainer(random_state=args.random_state)
            
            # Add different model types
            trainer.add_model(RandomForestModel(random_state=args.random_state))
            trainer.add_model(LogisticRegressionModel(random_state=args.random_state))
            trainer.add_model(NeuralNetworkModel(random_state=args.random_state))
            
            # Add advanced models if available
            try:
                trainer.add_model(XGBoostModel(random_state=args.random_state))
            except ImportError:
                logger.warning("XGBoost not available, skipping")
            
            try:
                trainer.add_model(LightGBMModel(random_state=args.random_state))
            except ImportError:
                logger.warning("LightGBM not available, skipping")
            
            # For demonstration, create sample data
            # In a real scenario, you'd use the engineered features
            logger.info("Creating sample training data for demonstration...")
            
            # This is just for demonstration - replace with real data
            import pandas as pd
            import numpy as np
            
            # Create sample features and target
            np.random.seed(args.random_state)
            n_samples = 1000
            n_features = 20
            
            X_sample = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            )
            
            # Create sample target (home team win/loss)
            y_sample = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
            
            # Train all models
            logger.info("Training all models...")
            results = trainer.train_all_models(
                X_sample, y_sample, 
                test_size=args.test_size
            )
            
            # Display results
            comparison_df = trainer.get_model_comparison()
            logger.info("\nModel Comparison Results:")
            logger.info(comparison_df.to_string(index=False))
            
            # Save best model
            if trainer.best_model:
                model_path = f"./models/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                trainer.save_best_model(model_path)
                logger.info(f"Best model saved to: {model_path}")
        
        if args.mode in ['predict', 'full']:
            # Prediction Phase
            logger.info("=== Starting Prediction Phase ===")
            
            if 'trainer' in locals() and trainer.best_model:
                # Create sample prediction data
                X_pred = pd.DataFrame(
                    np.random.randn(10, n_features),
                    columns=[f'feature_{i}' for i in range(n_features)]
                )
                
                # Make predictions
                predictions, probabilities = trainer.predict_with_best_model(X_pred)
                
                logger.info("Sample Predictions:")
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    outcome = "Home Win" if pred == 1 else "Away Win"
                    confidence = max(prob) if prob is not None else "N/A"
                    logger.info(f"Game {i+1}: {outcome} (Confidence: {confidence:.3f})")
            else:
                logger.warning("No trained model available for predictions")
        
        logger.info("=== Pipeline Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

def demo_data_collection():
    """Demonstrate data collection functionality"""
    logger = get_logger("demo")
    
    logger.info("Demonstrating data collection...")
    
    try:
        collector = NFLDataCollector()
        
        # Collect teams
        logger.info("Collecting team information...")
        teams = collector.collect_teams()
        logger.info(f"Collected data for {len(teams.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', []))} teams")
        
        # Collect current scoreboard
        logger.info("Collecting current scoreboard...")
        scoreboard = collector.collect_scoreboard(2024, 2)  # 2024 regular season
        logger.info("Scoreboard data collected successfully")
        
        # Collect schedule
        logger.info("Collecting schedule...")
        schedule = collector.collect_schedule(2024, 2)
        logger.info("Schedule data collected successfully")
        
        logger.info("Data collection demonstration completed")
        
    except Exception as e:
        logger.error(f"Data collection demonstration failed: {e}")

if __name__ == "__main__":
    main() 