"""
Feature Engineering for NFL Game Prediction
Transforms raw ESPN API data into ML-ready features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_selection import SelectKBest, f_classif
except ImportError:
    print("scikit-learn not available. Install with: pip install scikit-learn")
    StandardScaler = None
    LabelEncoder = None
    SelectKBest = None
    f_classif = None

logger = logging.getLogger(__name__)

class NFLFeatureEngineer:
    """Engineers features from NFL data for machine learning models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def extract_game_features(self, game_data: Dict) -> Dict[str, Any]:
        """Extract basic game features from game data"""
        features = {}
        
        try:
            # Basic game info
            features['event_id'] = game_data.get('id')
            features['date'] = game_data.get('date')
            features['season_type'] = game_data.get('season', {}).get('type')
            features['week'] = game_data.get('week', {}).get('number')
            
            # Venue info
            venue = game_data.get('competitions', [{}])[0].get('venue', {})
            features['venue_id'] = venue.get('id')
            features['venue_capacity'] = venue.get('capacity')
            features['venue_indoor'] = venue.get('indoor', False)
            
            # Weather info if available
            weather = game_data.get('weather', {})
            features['temperature'] = weather.get('temperature')
            features['wind_speed'] = weather.get('windSpeed')
            features['humidity'] = weather.get('humidity')
            features['precipitation'] = weather.get('precipitation')
            
        except Exception as e:
            logger.error(f"Error extracting game features: {e}")
            
        return features
    
    def extract_team_features(self, team_data: Dict, is_home: bool = True) -> Dict[str, Any]:
        """Extract team-specific features"""
        features = {}
        prefix = "home" if is_home else "away"
        
        try:
            # Team basic info
            features[f'{prefix}_team_id'] = team_data.get('id')
            features[f'{prefix}_team_name'] = team_data.get('name')
            features[f'{prefix}_team_abbrev'] = team_data.get('abbreviation')
            
            # Team record
            record = team_data.get('record', [{}])[0] if team_data.get('record') else {}
            features[f'{prefix}_wins'] = record.get('wins', 0)
            features[f'{prefix}_losses'] = record.get('losses', 0)
            features[f'{prefix}_ties'] = record.get('ties', 0)
            features[f'{prefix}_win_pct'] = record.get('percentage', 0)
            
            # Team stats if available
            stats = team_data.get('statistics', [])
            if stats:
                for stat in stats:
                    stat_name = stat.get('name', '')
                    stat_value = stat.get('displayValue', '0')
                    
                    # Convert to numeric if possible
                    try:
                        stat_value = float(stat_value)
                    except (ValueError, TypeError):
                        stat_value = 0
                    
                    features[f'{prefix}_{stat_name}'] = stat_value
            
            # Team ranking
            ranking = team_data.get('rank', 0)
            features[f'{prefix}_rank'] = ranking
            
        except Exception as e:
            logger.error(f"Error extracting team features: {e}")
            
        return features
    
    def extract_player_features(self, player_data: Dict, team_prefix: str) -> Dict[str, Any]:
        """Extract player-specific features"""
        features = {}
        
        try:
            # Basic player info
            features[f'{team_prefix}_player_id'] = player_data.get('id')
            features[f'{team_prefix}_player_name'] = player_data.get('fullName')
            features[f'{team_prefix}_player_position'] = player_data.get('position', {}).get('abbreviation')
            
            # Player stats
            stats = player_data.get('statistics', [])
            if stats:
                for stat in stats:
                    stat_name = stat.get('name', '')
                    stat_value = stat.get('displayValue', '0')
                    
                    try:
                        stat_value = float(stat_value)
                    except (ValueError, TypeError):
                        stat_value = 0
                    
                    features[f'{team_prefix}_{stat_name}'] = stat_value
                    
        except Exception as e:
            logger.error(f"Error extracting player features: {e}")
            
        return features
    
    def create_game_features(self, game_data: Dict, home_team: Dict, away_team: Dict) -> Dict[str, Any]:
        """Create comprehensive game features combining all data sources"""
        features = {}
        
        try:
            # Game features
            game_features = self.extract_game_features(game_data)
            features.update(game_features)
            
            # Team features
            home_features = self.extract_team_features(home_team, is_home=True)
            away_features = self.extract_team_features(away_team, is_home=False)
            features.update(home_features)
            features.update(away_features)
            
            # Derived features
            features.update(self._create_derived_features(features))
            
        except Exception as e:
            logger.error(f"Error creating game features: {e}")
            
        return features
    
    def _create_derived_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Create derived features from existing features"""
        derived = {}
        
        try:
            # Win percentage differences
            if 'home_win_pct' in features and 'away_win_pct' in features:
                derived['win_pct_diff'] = features['home_win_pct'] - features['away_win_pct']
                derived['win_pct_sum'] = features['home_win_pct'] + features['away_win_pct']
            
            # Record differences
            if 'home_wins' in features and 'away_wins' in features:
                derived['wins_diff'] = features['home_wins'] - features['away_wins']
                derived['losses_diff'] = features['home_losses'] - features['away_losses']
            
            # Team strength indicators
            if 'home_rank' in features and 'away_rank' in features:
                derived['rank_diff'] = features['away_rank'] - features['home_rank']  # Lower rank = better
                derived['rank_sum'] = features['home_rank'] + features['away_rank']
            
            # Venue advantage
            if 'venue_capacity' in features:
                derived['venue_advantage'] = 1 if features.get('venue_indoor', False) else 0
            
            # Season timing
            if 'week' in features:
                derived['late_season'] = 1 if features['week'] > 12 else 0
                derived['playoff_push'] = 1 if features['week'] > 14 else 0
            
        except Exception as e:
            logger.error(f"Error creating derived features: {e}")
            
        return derived
    
    def create_historical_features(self, historical_data: List[Dict], lookback_games: int = 5) -> pd.DataFrame:
        """Create features from historical game data"""
        all_features = []
        
        try:
            for game in historical_data:
                features = self.create_game_features(
                    game['game_data'],
                    game['home_team'],
                    game['away_team']
                )
                
                # Add historical performance features
                historical_features = self._add_historical_performance(
                    game, historical_data, lookback_games
                )
                features.update(historical_features)
                
                all_features.append(features)
                
        except Exception as e:
            logger.error(f"Error creating historical features: {e}")
            
        return pd.DataFrame(all_features)
    
    def _add_historical_performance(self, current_game: Dict, historical_data: List[Dict], lookback: int) -> Dict[str, Any]:
        """Add historical performance features for teams"""
        features = {}
        
        try:
            home_team_id = current_game['home_team']['id']
            away_team_id = current_game['away_team']['id']
            current_date = current_game['game_data']['date']
            
            # Get recent games for both teams
            home_recent = self._get_recent_team_games(historical_data, home_team_id, current_date, lookback)
            away_recent = self._get_recent_team_games(historical_data, away_team_id, current_date, lookback)
            
            # Calculate recent performance metrics
            features.update(self._calculate_recent_performance(home_recent, 'home_recent'))
            features.update(self._calculate_recent_performance(away_recent, 'away_recent'))
            
        except Exception as e:
            logger.error(f"Error adding historical performance: {e}")
            
        return features
    
    def _get_recent_team_games(self, historical_data: List[Dict], team_id: int, current_date: str, lookback: int) -> List[Dict]:
        """Get recent games for a specific team"""
        recent_games = []
        
        try:
            current_dt = datetime.fromisoformat(current_date.replace('Z', '+00:00'))
            
            for game in historical_data:
                game_date = datetime.fromisoformat(game['game_data']['date'].replace('Z', '+00:00'))
                
                if game_date < current_dt:
                    # Check if team played in this game
                    if (game['home_team']['id'] == team_id or 
                        game['away_team']['id'] == team_id):
                        recent_games.append(game)
                        
                        if len(recent_games) >= lookback:
                            break
                            
        except Exception as e:
            logger.error(f"Error getting recent team games: {e}")
            
        return recent_games
    
    def _calculate_recent_performance(self, recent_games: List[Dict], prefix: str) -> Dict[str, Any]:
        """Calculate recent performance metrics for a team"""
        features = {}
        
        try:
            if not recent_games:
                features[f'{prefix}_avg_points'] = 0
                features[f'{prefix}_avg_points_allowed'] = 0
                features[f'{prefix}_wins'] = 0
                features[f'{prefix}_losses'] = 0
                features[f'{prefix}_win_pct'] = 0
                return features
            
            total_points = 0
            total_points_allowed = 0
            wins = 0
            losses = 0
            
            for game in recent_games:
                # Determine if team is home or away
                is_home = game['home_team']['id'] == game.get('team_id')
                team_data = game['home_team'] if is_home else game['away_team']
                opponent_data = game['away_team'] if is_home else game['home_team']
                
                # Get scores
                team_score = team_data.get('score', 0)
                opponent_score = opponent_data.get('score', 0)
                
                total_points += team_score
                total_points_allowed += opponent_score
                
                if team_score > opponent_score:
                    wins += 1
                else:
                    losses += 1
            
            # Calculate averages
            num_games = len(recent_games)
            features[f'{prefix}_avg_points'] = total_points / num_games if num_games > 0 else 0
            features[f'{prefix}_avg_points_allowed'] = total_points_allowed / num_games if num_games > 0 else 0
            features[f'{prefix}_wins'] = wins
            features[f'{prefix}_losses'] = losses
            features[f'{prefix}_win_pct'] = wins / num_games if num_games > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating recent performance: {e}")
            
        return features
    
    def prepare_features_for_ml(self, features_df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features for machine learning models"""
        try:
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
            # Encode categorical variables
            features_df = self._encode_categorical_variables(features_df)
            
            # Scale numerical features
            features_df = self._scale_numerical_features(features_df)
            
            # Feature selection
            if target_column and target_column in features_df.columns:
                features_df = self._select_features(features_df, features_df[target_column])
            
            # Store feature names
            self.feature_names = features_df.columns.tolist()
            
            # Split features and target
            if target_column and target_column in features_df.columns:
                X = features_df.drop(columns=[target_column])
                y = features_df[target_column]
                return X, y
            else:
                return features_df, None
                
        except Exception as e:
            logger.error(f"Error preparing features for ML: {e}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
                
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    # Handle new categories
                    df[col] = self.label_encoders[col].transform(df[col])
                    
        except Exception as e:
            logger.error(f"Error encoding categorical variables: {e}")
            
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) > 0:
                df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
                
        except Exception as e:
            logger.error(f"Error scaling numerical features: {e}")
            
        return df
    
    def _select_features(self, df: pd.DataFrame, target: pd.Series, k: int = 50) -> pd.DataFrame:
        """Select top k features using statistical tests"""
        try:
            if k >= len(df.columns):
                return df
            
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(df, target)
            
            selected_features = df.columns[selector.get_support()].tolist()
            return df[selected_features]
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return df
    
    def get_feature_importance(self, model, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance from a trained model"""
        try:
            if feature_names is None:
                feature_names = self.feature_names
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                logger.warning("Model doesn't have feature importance or coefficients")
                return pd.DataFrame()
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame() 