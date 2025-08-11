"""
Machine Learning Models for NFL Game Prediction
Implements various ML algorithms and ensemble methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import pickle
from pathlib import Path
from datetime import datetime

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

# Model evaluation
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

logger = logging.getLogger(__name__)

class NFLPredictionModel:
    """Base class for NFL prediction models"""
    
    def __init__(self, model_name: str, random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.classes_ = None
        self.training_history = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the model"""
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Model doesn't support probability predictions")
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'classes_': self.classes_,
            'training_history': self.training_history,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.classes_ = model_data['classes_']
        self.training_history = model_data['training_history']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")

class RandomForestModel(NFLPredictionModel):
    """Random Forest model for NFL prediction"""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__("RandomForest", random_state)
        # Extract specific parameters to avoid conflicts
        n_estimators = kwargs.pop('n_estimators', 100)
        max_depth = kwargs.pop('max_depth', None)
        min_samples_split = kwargs.pop('min_samples_split', 2)
        min_samples_leaf = kwargs.pop('min_samples_leaf', 1)
        
        self.model = RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            **kwargs
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the Random Forest model"""
        logger.info(f"Training {self.model_name} model...")
        
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True
        
        logger.info(f"{self.model_name} model training completed")

class XGBoostModel(NFLPredictionModel):
    """XGBoost model for NFL prediction"""
    
    def __init__(self, random_state: int = 42, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
        
        super().__init__("XGBoost", random_state)
        self.model = xgb.XGBClassifier(
            random_state=random_state,
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            **kwargs
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the XGBoost model"""
        logger.info(f"Training {self.model_name} model...")
        
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True
        
        logger.info(f"{self.model_name} model training completed")

class LightGBMModel(NFLPredictionModel):
    """LightGBM model for NFL prediction"""
    
    def __init__(self, random_state: int = 42, **kwargs):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
        
        super().__init__("LightGBM", random_state)
        self.model = lgb.LGBMClassifier(
            random_state=random_state,
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', -1),
            learning_rate=kwargs.get('learning_rate', 0.1),
            **kwargs
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the LightGBM model"""
        logger.info(f"Training {self.model_name} model...")
        
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True
        
        logger.info(f"{self.model_name} model training completed")

class LogisticRegressionModel(NFLPredictionModel):
    """Logistic Regression model for NFL prediction"""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__("LogisticRegression", random_state)
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=kwargs.get('max_iter', 1000),
            C=kwargs.get('C', 1.0),
            **kwargs
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the Logistic Regression model"""
        logger.info(f"Training {self.model_name} model...")
        
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True
        
        logger.info(f"{self.model_name} model training completed")

class NeuralNetworkModel(NFLPredictionModel):
    """Neural Network model for NFL prediction"""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__("NeuralNetwork", random_state)
        self.model = MLPClassifier(
            random_state=random_state,
            hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100, 50)),
            max_iter=kwargs.get('max_iter', 500),
            learning_rate_init=kwargs.get('learning_rate_init', 0.001),
            **kwargs
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the Neural Network model"""
        logger.info(f"Training {self.model_name} model...")
        
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True
        
        logger.info(f"{self.model_name} model training completed")

class EnsembleModel(NFLPredictionModel):
    """Ensemble model combining multiple base models"""
    
    def __init__(self, base_models: List[NFLPredictionModel], random_state: int = 42):
        super().__init__("Ensemble", random_state)
        self.base_models = base_models
        self.model = None  # Will be set during training
        self.model_weights = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the ensemble model"""
        logger.info(f"Training {self.model_name} model with {len(self.base_models)} base models...")
        
        self.feature_names = X.columns.tolist()
        
        # Train all base models
        for model in self.base_models:
            model.train(X, y, **kwargs)
        
        # Simple voting ensemble (can be enhanced with weighted voting)
        self.is_trained = True
        logger.info(f"{self.model_name} model training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Get predictions from all base models
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Simple majority voting
        predictions_array = np.array(predictions)
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions_array
        )
        
        return ensemble_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble prediction probabilities"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Average probabilities from all base models
        probabilities = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)
                probabilities.append(prob)
        
        if probabilities:
            return np.mean(probabilities, axis=0)
        else:
            raise NotImplementedError("Base models don't support probability predictions")

class ModelTrainer:
    """Handles model training, evaluation, and comparison"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.training_results = {}
        self.best_model = None
        
    def add_model(self, model: NFLPredictionModel, name: Optional[str] = None):
        """Add a model to the trainer"""
        model_name = name or model.model_name
        self.models[model_name] = model
        logger.info(f"Added model: {model_name}")
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, **kwargs):
        """Train all models and evaluate performance"""
        logger.info("Starting training for all models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Train model
                model.train(X_train, y_train, **kwargs)
                
                # Evaluate model
                train_score = model.model.score(X_train, y_train)
                test_score = model.model.score(X_test, y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model.model, X_train, y_train, cv=5)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                results[name] = {
                    'model': model,
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - Train: {train_score:.4f}, Test: {test_score:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.training_results = results
        
        # Find best model
        self._find_best_model()
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def _find_best_model(self):
        """Find the best performing model based on test score"""
        valid_results = {k: v for k, v in self.training_results.items() if 'error' not in v}
        
        if valid_results:
            best_name = max(valid_results.keys(), 
                          key=lambda x: valid_results[x]['test_score'])
            self.best_model = valid_results[best_name]['model']
            logger.info(f"Best model: {best_name} with test score: {valid_results[best_name]['test_score']:.4f}")
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get a comparison DataFrame of all models"""
        if not self.training_results:
            return pd.DataFrame()
        
        comparison_data = []
        for name, results in self.training_results.items():
            if 'error' not in results:
                comparison_data.append({
                    'Model': name,
                    'Train Score': results['train_score'],
                    'Test Score': results['test_score'],
                    'CV Mean': results['cv_mean'],
                    'CV Std': results['cv_std'],
                    'Accuracy': results['metrics']['accuracy'],
                    'Precision': results['metrics']['precision'],
                    'Recall': results['metrics']['recall'],
                    'F1 Score': results['metrics']['f1'],
                    'ROC AUC': results['metrics'].get('roc_auc', 'N/A')
                })
        
        return pd.DataFrame(comparison_data).sort_values('Test Score', ascending=False)
    
    def save_best_model(self, filepath: str):
        """Save the best performing model"""
        if self.best_model:
            self.best_model.save_model(filepath)
        else:
            logger.warning("No best model available to save")
    
    def load_model(self, name: str, filepath: str):
        """Load a specific model"""
        if name in self.models:
            self.models[name].load_model(filepath)
            logger.info(f"Loaded model {name} from {filepath}")
        else:
            logger.error(f"Model {name} not found in trainer")
    
    def predict_with_best_model(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using the best model"""
        if not self.best_model:
            raise RuntimeError("No best model available. Train models first.")
        
        predictions = self.best_model.predict(X)
        probabilities = None
        
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X)
        
        return predictions, probabilities 