"""
ESPN API Client for NFL data collection
Handles all API requests to ESPN endpoints with rate limiting and error handling
"""

import asyncio
import aiohttp
import requests
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ESPNConfig:
    """Configuration for ESPN API client"""
    base_url: str = "https://site.api.espn.com/apis/site/v2"
    core_api_url: str = "https://sports.core.api.espn.com/v2"
    cdn_api_url: str = "https://cdn.espn.com/core"
    rate_limit: int = 10  # requests per second
    timeout: int = 30
    max_retries: int = 3

class ESPNClient:
    """Client for interacting with ESPN NFL API endpoints"""
    
    def __init__(self, config: Optional[ESPNConfig] = None):
        self.config = config or ESPNConfig()
        self.session = None
        self.last_request_time = 0
        self.request_count = 0
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _rate_limit(self):
        """Simple rate limiting for synchronous requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.config.rate_limit
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _async_rate_limit(self):
        """Async rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.config.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make synchronous HTTP request with rate limiting"""
        self._rate_limit()
        
        try:
            response = requests.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
    
    async def _make_async_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make asynchronous HTTP request with rate limiting"""
        await self._async_rate_limit()
        
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Async request failed for {url}: {e}")
            raise
    
    # Scoreboard and Schedule Methods
    def get_scoreboard(self, year: int, season_type: int = 2, week: Optional[int] = None) -> Dict:
        """Get NFL scoreboard data"""
        url = f"{self.config.base_url}/sports/football/nfl/scoreboard"
        params = {"dates": year}
        
        if season_type:
            params["seasontype"] = season_type
        if week:
            params["week"] = week
            
        return self._make_request(url, params)
    
    def get_schedule(self, year: int, season_type: int = 2, week: Optional[int] = None) -> Dict:
        """Get NFL schedule data"""
        url = f"{self.config.cdn_api_url}/nfl/schedule"
        params = {"xhr": 1, "year": year}
        
        if season_type:
            params["seasontype"] = season_type
        if week:
            params["week"] = week
            
        return self._make_request(url, params)
    
    # Team and Player Methods
    def get_teams(self) -> Dict:
        """Get list of all NFL teams"""
        url = f"{self.config.core_api_url}/sports/football/leagues/nfl/teams"
        params = {"limit": 32}
        return self._make_request(url, params)
    
    def get_athletes(self, active_only: bool = True) -> Dict:
        """Get list of NFL athletes"""
        url = f"{self.config.core_api_url}/sports/football/leagues/nfl/athletes"
        params = {"limit": 1000, "active": active_only}
        return self._make_request(url, params)
    
    def get_team_stats(self, team_id: int, year: int) -> Dict:
        """Get team statistics for a specific year"""
        url = f"{self.config.core_api_url}/sports/football/leagues/nfl/seasons/{year}/teams/{team_id}/statistics"
        return self._make_request(url)
    
    # Game Details Methods
    def get_game_details(self, event_id: int) -> Dict:
        """Get detailed game information"""
        url = f"{self.config.cdn_api_url}/nfl/game"
        params = {"xhr": 1, "gameId": event_id}
        return self._make_request(url, params)
    
    def get_boxscore(self, event_id: int) -> Dict:
        """Get game boxscore"""
        url = f"{self.config.cdn_api_url}/nfl/boxscore"
        params = {"xhr": 1, "gameId": event_id}
        return self._make_request(url, params)
    
    def get_play_by_play(self, event_id: int) -> Dict:
        """Get play-by-play data"""
        url = f"{self.config.cdn_api_url}/nfl/playbyplay"
        params = {"xhr": 1, "gameId": event_id}
        return self._make_request(url, params)
    
    # Odds and Predictions
    def get_win_probabilities(self, event_id: int) -> Dict:
        """Get win probabilities for a game"""
        url = f"{self.config.core_api_url}/sports/football/leagues/nfl/events/{event_id}/competitions/{event_id}/probabilities"
        params = {"limit": 200}
        return self._make_request(url, params)
    
    def get_odds(self, event_id: int) -> Dict:
        """Get betting odds for a game"""
        url = f"{self.config.core_api_url}/sports/football/leagues/nfl/events/{event_id}/competitions/{event_id}/odds"
        return self._make_request(url)
    
    def get_predictor(self, event_id: int) -> Dict:
        """Get matchup quality and game projections"""
        url = f"{self.config.core_api_url}/sports/football/leagues/nfl/events/{event_id}/competitions/{event_id}/predictor"
        return self._make_request(url)
    
    # News and Information
    def get_news(self, limit: int = 50, team_id: Optional[int] = None) -> Dict:
        """Get NFL news"""
        url = f"{self.config.base_url}/sports/football/nfl/news"
        params = {"limit": limit}
        
        if team_id:
            params["team"] = team_id
            
        return self._make_request(url, params)
    
    def get_standings(self) -> Dict:
        """Get NFL standings"""
        url = f"{self.config.cdn_api_url}/nfl/standings"
        params = {"xhr": 1}
        return self._make_request(url, params)
    
    # Historical Data Methods
    def get_historical_scoreboard(self, year: int, season_type: int = 2) -> Dict:
        """Get historical scoreboard data"""
        return self.get_scoreboard(year, season_type)
    
    # Utility Methods
    def get_season_info(self) -> Dict:
        """Get current season information"""
        url = f"{self.config.core_api_url}/sports/football/leagues/nfl/seasons"
        params = {"limit": 100}
        return self._make_request(url, params)
    
    def get_positions(self) -> Dict:
        """Get list of NFL positions"""
        url = f"{self.config.core_api_url}/sports/football/leagues/nfl/positions"
        params = {"limit": 75}
        return self._make_request(url, params)
    
    def get_venues(self) -> Dict:
        """Get list of NFL venues"""
        url = f"{self.config.core_api_url}/sports/football/leagues/nfl/venues"
        params = {"limit": 700}
        return self._make_request(url, params) 