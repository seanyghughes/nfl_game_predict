# NFL Game Prediction ML Project

This project uses machine learning to predict the outcomes of NFL games using data from ESPN's API endpoints.

## Project Structure

```
nfl_game_predict/
├── data/                   # Data storage and caching
├── models/                 # Trained ML models
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code
│   ├── api/               # ESPN API integration
│   ├── data/              # Data processing and feature engineering
│   ├── ml/                # Machine learning models
│   └── utils/             # Utility functions
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── main.py                # Main execution script
```

## Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the project:**
   ```bash
   python main.py
   ```

## ESPN API Endpoints Used

This project leverages the following ESPN API endpoints for data collection:

- **Scoreboard**: Get current and historical game data
- **Team Statistics**: Player and team performance metrics
- **Game Details**: Play-by-play, box scores, and game statistics
- **Odds and Predictions**: Betting odds and win probabilities
- **Player Stats**: Individual player performance data

## Features

- Automated data collection from ESPN APIs
- Feature engineering for ML models
- Multiple ML algorithms (Random Forest, XGBoost, Neural Networks)
- Model evaluation and comparison
- Real-time prediction capabilities
- Historical performance analysis

## Data Sources

- ESPN NFL API endpoints
- Historical game results
- Player statistics
- Team performance metrics
- Weather data (optional)
- Injury reports (optional)

## Model Performance

The models are evaluated using:
- Accuracy metrics
- Precision/Recall
- F1 Score
- ROC-AUC curves
- Cross-validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details 