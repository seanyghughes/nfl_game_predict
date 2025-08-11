"""
Data Collector for NFL game data
Orchestrates data collection from ESPN APIs and stores data locally
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
from pathlib import Path

from src.api.espn_client import ESPNClient, ESPNConfig

logger = logging.getLogger(__name__)

class NFLDataCollector:
    """Collects and stores NFL data from ESPN APIs"""
    
    def __init__(self, data_dir: str = "./data", cache_dir: str = "./data/cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = ESPNClient()
        
    def _get_cache_path(self, data_type: str, identifier: str) -> Path:
        """Get cache file path for a specific data type and identifier"""
        return self.cache_dir / f"{data_type}_{identifier}.json"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached data is still valid"""
        if not cache_path.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age.total_seconds() < (max_age_hours * 3600)
    
    def _save_to_cache(self, data: Dict, data_type: str, identifier: str):
        """Save data to cache file"""
        cache_path = self._get_cache_path(data_type, identifier)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {data_type} data to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache for {data_type}: {e}")
    
    def _load_from_cache(self, data_type: str, identifier: str) -> Optional[Dict]:
        """Load data from cache if valid"""
        cache_path = self._get_cache_path(data_type, identifier)
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded {data_type} data from cache: {cache_path}")
                return data
            except Exception as e:
                logger.error(f"Failed to load cache for {data_type}: {e}")
        return None
    
    def collect_teams(self, use_cache: bool = True) -> Dict:
        """Collect team information"""
        if use_cache:
            cached_data = self._load_from_cache("teams", "all")
            if cached_data:
                return cached_data
        
        try:
            data = self.client.get_teams()
            self._save_to_cache(data, "teams", "all")
            return data
        except Exception as e:
            logger.error(f"Failed to collect teams data: {e}")
            raise
    
    def collect_athletes(self, use_cache: bool = True) -> Dict:
        """Collect athlete information"""
        if use_cache:
            cached_data = self._load_from_cache("athletes", "all")
            if cached_data:
                return cached_data
        
        try:
            data = self.client.get_athletes(active_only=True)
            self._save_to_cache(data, "athletes", "all")
            return data
        except Exception as e:
            logger.error(f"Failed to collect athletes data: {e}")
            raise
    
    def collect_scoreboard(self, year: int, season_type: int = 2, week: Optional[int] = None, use_cache: bool = True) -> Dict:
        """Collect scoreboard data for a specific year/week"""
        identifier = f"{year}_{season_type}_{week or 'all'}"
        
        if use_cache:
            cached_data = self._load_from_cache("scoreboard", identifier)
            if cached_data:
                return cached_data
        
        try:
            data = self.client.get_scoreboard(year, season_type, week)
            self._save_to_cache(data, "scoreboard", identifier)
            return data
        except Exception as e:
            logger.error(f"Failed to collect scoreboard data: {e}")
            raise
    
    def collect_schedule(self, year: int, season_type: int = 2, week: Optional[int] = None, use_cache: bool = True) -> Dict:
        """Collect schedule data for a specific year/week"""
        identifier = f"{year}_{season_type}_{week or 'all'}"
        
        if use_cache:
            cached_data = self._load_from_cache("schedule", identifier)
            if cached_data:
                return cached_data
        
        try:
            data = self.client.get_schedule(year, season_type, week)
            self._save_to_cache(data, "schedule", identifier)
            return data
        except Exception as e:
            logger.error(f"Failed to collect schedule data: {e}")
            raise
    
    def collect_game_details(self, event_id: int, use_cache: bool = True) -> Dict:
        """Collect detailed game information"""
        if use_cache:
            cached_data = self._load_from_cache("game_details", str(event_id))
            if cached_data:
                return cached_data
        
        try:
            data = self.client.get_game_details(event_id)
            self._save_to_cache(data, "game_details", str(event_id))
            return data
        except Exception as e:
            logger.error(f"Failed to collect game details for event {event_id}: {e}")
            raise
    
    def collect_boxscore(self, event_id: int, use_cache: bool = True) -> Dict:
        """Collect game boxscore"""
        if use_cache:
            cached_data = self._load_from_cache("boxscore", str(event_id))
            if cached_data:
                return cached_data
        
        try:
            data = self.client.get_boxscore(event_id)
            self._save_to_cache(data, "boxscore", str(event_id))
            return data
        except Exception as e:
            logger.error(f"Failed to collect boxscore for event {event_id}: {e}")
            raise
    
    def collect_play_by_play(self, event_id: int, use_cache: bool = True) -> Dict:
        """Collect play-by-play data"""
        if use_cache:
            cached_data = self._load_from_cache("play_by_play", str(event_id))
            if cached_data:
                return cached_data
        
        try:
            data = self.client.get_play_by_play(event_id)
            self._save_to_cache(data, "play_by_play", str(event_id))
            return data
        except Exception as e:
            logger.error(f"Failed to collect play-by-play for event {event_id}: {e}")
            raise
    
    def collect_team_stats(self, team_id: int, year: int, use_cache: bool = True) -> Dict:
        """Collect team statistics"""
        identifier = f"{team_id}_{year}"
        
        if use_cache:
            cached_data = self._load_from_cache("team_stats", identifier)
            if cached_data:
                return cached_data
        
        try:
            data = self.client.get_team_stats(team_id, year)
            self._save_to_cache(data, "team_stats", identifier)
            return data
        except Exception as e:
            logger.error(f"Failed to collect team stats for team {team_id}, year {year}: {e}")
            raise
    
    def collect_odds_and_predictions(self, event_id: int, use_cache: bool = True) -> Dict:
        """Collect odds and prediction data for a game"""
        if use_cache:
            cached_data = self._load_from_cache("odds_predictions", str(event_id))
            if cached_data:
                return cached_data
        
        try:
            # Collect multiple types of prediction data
            data = {
                "win_probabilities": self.client.get_win_probabilities(event_id),
                "odds": self.client.get_odds(event_id),
                "predictor": self.client.get_predictor(event_id)
            }
            self._save_to_cache(data, "odds_predictions", str(event_id))
            return data
        except Exception as e:
            logger.error(f"Failed to collect odds/predictions for event {event_id}: {e}")
            raise
    
    def collect_historical_data(self, start_year: int, end_year: int, season_type: int = 2) -> Dict:
        """Collect historical data for multiple years"""
        historical_data = {}
        
        for year in range(start_year, end_year + 1):
            try:
                logger.info(f"Collecting historical data for year {year}")
                historical_data[year] = {
                    "scoreboard": self.collect_scoreboard(year, season_type, use_cache=True),
                    "schedule": self.collect_schedule(year, season_type, use_cache=True),
                    "standings": self.client.get_standings()
                }
            except Exception as e:
                logger.error(f"Failed to collect historical data for year {year}: {e}")
                continue
        
        return historical_data
    
    def collect_current_season_data(self, year: Optional[int] = None) -> Dict:
        """Collect current season data"""
        if year is None:
            year = datetime.now().year
        
        try:
            return {
                "teams": self.collect_teams(),
                "athletes": self.collect_athletes(),
                "scoreboard": self.collect_scoreboard(year, 2),  # Regular season
                "schedule": self.collect_schedule(year, 2),
                "standings": self.client.get_standings(),
                "season_info": self.client.get_season_info()
            }
        except Exception as e:
            logger.error(f"Failed to collect current season data: {e}")
            raise
    
    def export_to_csv(self, data: Dict, filename: str, output_dir: Optional[str] = None):
        """Export data to CSV format"""
        output_path = Path(output_dir or self.processed_dir) / f"{filename}.csv"
        
        try:
            # Convert to DataFrame and save
            df = pd.json_normalize(data)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported data to CSV: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export data to CSV: {e}")
            raise
    
    def export_to_pickle(self, data: Any, filename: str, output_dir: Optional[str] = None):
        """Export data to pickle format"""
        output_path = Path(output_dir or self.processed_dir) / f"{filename}.pkl"
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Exported data to pickle: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export data to pickle: {e}")
            raise
    
    def clear_cache(self, data_type: Optional[str] = None):
        """Clear cache files"""
        if data_type:
            # Clear specific data type cache
            pattern = f"{data_type}_*.json"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                logger.info(f"Cleared cache file: {cache_file}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                logger.info(f"Cleared cache file: {cache_file}")
        
        logger.info("Cache clearing completed") 