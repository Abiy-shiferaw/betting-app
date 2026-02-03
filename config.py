"""
Configuration file for NBA Betting AI System — Parlay Edition v3
Move API keys to environment variables for production!
"""
import os

# API Keys - use environment variables in production
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "abba655dc7dfe42797d0dcda372eb86d")

# API Endpoints
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Odds API settings
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "decimal"

# File paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORICAL_DATA = os.path.join(DATA_DIR, "nba_games.csv")
ODDS_CACHE = os.path.join(DATA_DIR, "nba_odds_live.json")
MODEL_PATH = os.path.join(DATA_DIR, "model_v3.pkl")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions_v3.csv")
PARLAYS_PATH = os.path.join(DATA_DIR, "parlays_v3.csv")

# Team ID to Name mapping (NBA API team IDs)
TEAM_ID_TO_NAME = {
    1610612737: "Atlanta Hawks", 1610612738: "Boston Celtics",
    1610612751: "Brooklyn Nets", 1610612766: "Charlotte Hornets",
    1610612741: "Chicago Bulls", 1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks", 1610612743: "Denver Nuggets",
    1610612765: "Detroit Pistons", 1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets", 1610612754: "Indiana Pacers",
    1610612746: "Los Angeles Clippers", 1610612747: "Los Angeles Lakers",
    1610612763: "Memphis Grizzlies", 1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks", 1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans", 1610612752: "New York Knicks",
    1610612760: "Oklahoma City Thunder", 1610612753: "Orlando Magic",
    1610612755: "Philadelphia 76ers", 1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers", 1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs", 1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz", 1610612764: "Washington Wizards"
}

TEAM_NAME_TO_ID = {v: k for k, v in TEAM_ID_TO_NAME.items()}

# NBA Divisions (for parlay correlation analysis)
NBA_DIVISIONS = {
    "Atlantic": ["Boston Celtics", "Brooklyn Nets", "New York Knicks", "Philadelphia 76ers", "Toronto Raptors"],
    "Central": ["Chicago Bulls", "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers", "Milwaukee Bucks"],
    "Southeast": ["Atlanta Hawks", "Charlotte Hornets", "Miami Heat", "Orlando Magic", "Washington Wizards"],
    "Northwest": ["Denver Nuggets", "Minnesota Timberwolves", "Oklahoma City Thunder", "Portland Trail Blazers", "Utah Jazz"],
    "Pacific": ["Golden State Warriors", "Los Angeles Clippers", "Los Angeles Lakers", "Phoenix Suns", "Sacramento Kings"],
    "Southwest": ["Dallas Mavericks", "Houston Rockets", "Memphis Grizzlies", "New Orleans Pelicans", "San Antonio Spurs"],
}

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
ROLLING_WINDOW = 10  # Games for rolling averages

# Parlay parameters
PARLAY_MIN_CONFIDENCE = 60.0    # Minimum individual leg confidence %
PARLAY_MIN_EV = 0.0             # Minimum individual leg EV %
PARLAY_MAX_LEGS = 3             # Maximum legs in a parlay
PARLAY_MIN_WIN_RATE = 0.30      # Minimum combined parlay win probability
PARLAY_KELLY_FRACTION = 0.25    # Use quarter Kelly for parlays (conservative)

# Calibration
CALIBRATION_METHOD = "isotonic"  # "isotonic" or "sigmoid" (Platt)
MODEL_WEIGHT = 0.70              # 70% calibrated model, 30% market (up from 40/60)
