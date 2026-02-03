"""
Data Pipeline v3: Fetches odds, builds features from 30K+ historical games
Includes consistency features for parlay optimization
"""
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from config import *


def fetch_live_odds():
    """Fetch current NBA odds from The Odds API"""
    print("📡 Fetching live odds...")
    url = f"{ODDS_API_BASE}/sports/{SPORT}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        remaining = r.headers.get("x-requests-remaining", "?")
        print(f"✅ Fetched odds for {len(data)} games (API calls remaining: {remaining})")
        with open(ODDS_CACHE, "w") as f:
            json.dump(data, f, indent=2)
        return data
    except requests.RequestException as e:
        print(f"⚠️ Could not fetch live odds: {e}")
        # Try cached
        if os.path.exists(ODDS_CACHE):
            print("   Using cached odds data...")
            with open(ODDS_CACHE) as f:
                return json.load(f)
        return []


def parse_odds(odds_data):
    """Parse raw odds API data into structured format for each game"""
    games = []
    for game in odds_data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        if not home or not away:
            continue

        commence = game.get("commence_time", "")
        try:
            dt = datetime.strptime(commence, "%Y-%m-%dT%H:%M:%SZ")
            display_time = dt.strftime("%a, %b %d %H:%M")
            days_away = (dt - datetime.utcnow()).total_seconds() / 86400
        except ValueError:
            display_time = "TBD"
            days_away = 999

        # Extract best odds across all bookmakers
        h2h = {"home_odds": [], "away_odds": [], "home_books": [], "away_books": []}
        spreads = {"home_spread": [], "home_odds": [], "away_spread": [], "away_odds": [],
                   "home_books": [], "away_books": []}
        totals = {"line": [], "over_odds": [], "under_odds": [],
                  "over_books": [], "under_books": []}

        for bk in game.get("bookmakers", []):
            bk_name = bk.get("title", "")
            for mkt in bk.get("markets", []):
                key = mkt.get("key", "")
                outcomes = {o["name"]: o for o in mkt.get("outcomes", [])}

                if key == "h2h":
                    if home in outcomes:
                        h2h["home_odds"].append(outcomes[home]["price"])
                        h2h["home_books"].append(bk_name)
                    if away in outcomes:
                        h2h["away_odds"].append(outcomes[away]["price"])
                        h2h["away_books"].append(bk_name)

                elif key == "spreads":
                    if home in outcomes:
                        spreads["home_spread"].append(outcomes[home].get("point", 0))
                        spreads["home_odds"].append(outcomes[home]["price"])
                        spreads["home_books"].append(bk_name)
                    if away in outcomes:
                        spreads["away_spread"].append(outcomes[away].get("point", 0))
                        spreads["away_odds"].append(outcomes[away]["price"])
                        spreads["away_books"].append(bk_name)

                elif key == "totals":
                    if "Over" in outcomes:
                        totals["line"].append(outcomes["Over"].get("point", 0))
                        totals["over_odds"].append(outcomes["Over"]["price"])
                        totals["over_books"].append(bk_name)
                    if "Under" in outcomes:
                        totals["under_odds"].append(outcomes["Under"]["price"])
                        totals["under_books"].append(bk_name)

        g = {
            "home_team": home,
            "away_team": away,
            "game_time": display_time,
            "commence_time": commence,
            "days_until": round(days_away, 2),
        }

        # Best moneyline
        if h2h["home_odds"]:
            best_idx = np.argmax(h2h["home_odds"])
            g["ml_home_odds"] = h2h["home_odds"][best_idx]
            g["ml_home_book"] = h2h["home_books"][best_idx]
            g["ml_home_implied"] = 1 / h2h["home_odds"][best_idx]
        if h2h["away_odds"]:
            best_idx = np.argmax(h2h["away_odds"])
            g["ml_away_odds"] = h2h["away_odds"][best_idx]
            g["ml_away_book"] = h2h["away_books"][best_idx]
            g["ml_away_implied"] = 1 / h2h["away_odds"][best_idx]

        # Consensus moneyline (average across books) - more stable signal
        if h2h["home_odds"] and h2h["away_odds"]:
            g["ml_home_consensus"] = 1 / np.mean(h2h["home_odds"])
            g["ml_away_consensus"] = 1 / np.mean(h2h["away_odds"])
            # Normalize to sum to 1 (remove vig)
            total_prob = g["ml_home_consensus"] + g["ml_away_consensus"]
            g["ml_home_true_prob"] = g["ml_home_consensus"] / total_prob
            g["ml_away_true_prob"] = g["ml_away_consensus"] / total_prob

        # Best spread
        if spreads["home_spread"]:
            # Use consensus spread (median)
            g["spread_home"] = np.median(spreads["home_spread"])
            g["spread_away"] = -g["spread_home"]
            best_idx = np.argmax(spreads["home_odds"])
            g["spread_home_odds"] = spreads["home_odds"][best_idx]
            g["spread_home_book"] = spreads["home_books"][best_idx]
            if spreads["away_odds"]:
                best_idx = np.argmax(spreads["away_odds"])
                g["spread_away_odds"] = spreads["away_odds"][best_idx]
                g["spread_away_book"] = spreads["away_books"][best_idx]

        # Best totals
        if totals["line"]:
            g["total_line"] = np.median(totals["line"])
            if totals["over_odds"]:
                best_idx = np.argmax(totals["over_odds"])
                g["total_over_odds"] = totals["over_odds"][best_idx]
                g["total_over_book"] = totals["over_books"][best_idx]
            if totals["under_odds"]:
                best_idx = np.argmax(totals["under_odds"])
                g["total_under_odds"] = totals["under_odds"][best_idx]
                g["total_under_book"] = totals["under_books"][best_idx]

        games.append(g)

    return pd.DataFrame(games)


def load_historical_data():
    """Load and prepare the 26K+ historical games dataset"""
    print("📊 Loading historical data...")
    df = pd.read_csv(HISTORICAL_DATA)
    print(f"   Loaded {len(df)} games from {df['GAME_DATE_EST'].min()} to {df['GAME_DATE_EST'].max()}")

    # Map team IDs to names
    df["home_team"] = df["HOME_TEAM_ID"].map(TEAM_ID_TO_NAME)
    df["away_team"] = df["VISITOR_TEAM_ID"].map(TEAM_ID_TO_NAME)
    df["date"] = pd.to_datetime(df["GAME_DATE_EST"])
    df = df.sort_values("date").reset_index(drop=True)

    # Target: did home team win?
    df["home_win"] = df["HOME_TEAM_WINS"].astype(int)

    # Rename for clarity
    df = df.rename(columns={
        "PTS_home": "home_pts", "PTS_away": "away_pts",
        "FG_PCT_home": "home_fg_pct", "FG_PCT_away": "away_fg_pct",
        "FT_PCT_home": "home_ft_pct", "FT_PCT_away": "away_ft_pct",
        "FG3_PCT_home": "home_fg3_pct", "FG3_PCT_away": "away_fg3_pct",
        "AST_home": "home_ast", "AST_away": "away_ast",
        "REB_home": "home_reb", "REB_away": "away_reb",
    })

    return df


def build_features(df):
    """
    Build rolling/aggregated features for each game.
    Uses ONLY information available BEFORE the game (no data leakage).
    """
    print("🔧 Engineering features...")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # We'll compute rolling stats for each team
    teams = set(df["home_team"].dropna().unique()) | set(df["away_team"].dropna().unique())

    # Build a game log per team: for each game, whether they were home or away
    team_stats = {}
    for team in teams:
        home_games = df[df["home_team"] == team][["date", "home_pts", "away_pts", "home_win",
                                                     "home_fg_pct", "home_ft_pct", "home_fg3_pct",
                                                     "home_ast", "home_reb"]].copy()
        home_games.columns = ["date", "pts_for", "pts_against", "win",
                              "fg_pct", "ft_pct", "fg3_pct", "ast", "reb"]
        home_games["is_home"] = 1

        away_games = df[df["away_team"] == team][["date", "away_pts", "home_pts", "home_win",
                                                     "away_fg_pct", "away_ft_pct", "away_fg3_pct",
                                                     "away_ast", "away_reb"]].copy()
        away_games.columns = ["date", "pts_for", "pts_against", "win",
                              "fg_pct", "ft_pct", "fg3_pct", "ast", "reb"]
        away_games["win"] = 1 - away_games["win"]  # Flip: home_win=0 means away team won
        away_games["is_home"] = 0

        team_log = pd.concat([home_games, away_games]).sort_values("date").reset_index(drop=True)
        team_stats[team] = team_log

    # Now compute rolling features for each game in df
    W = ROLLING_WINDOW  # Window size

    feature_rows = []
    for idx, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        game_date = row["date"]

        if pd.isna(ht) or pd.isna(at):
            feature_rows.append({})
            continue

        features = {}

        for prefix, team in [("home", ht), ("away", at)]:
            tlog = team_stats.get(team)
            if tlog is None:
                continue
            # Only games BEFORE this date (no leakage)
            past = tlog[tlog["date"] < game_date].tail(W)

            if len(past) >= 3:  # Need at least 3 games
                features[f"{prefix}_win_pct"] = past["win"].mean()
                features[f"{prefix}_avg_pts"] = past["pts_for"].mean()
                features[f"{prefix}_avg_pts_allowed"] = past["pts_against"].mean()
                features[f"{prefix}_avg_fg_pct"] = past["fg_pct"].mean()
                features[f"{prefix}_avg_ft_pct"] = past["ft_pct"].mean()
                features[f"{prefix}_avg_fg3_pct"] = past["fg3_pct"].mean()
                features[f"{prefix}_avg_ast"] = past["ast"].mean()
                features[f"{prefix}_avg_reb"] = past["reb"].mean()
                features[f"{prefix}_net_rating"] = past["pts_for"].mean() - past["pts_against"].mean()

                # Streak: count consecutive wins/losses from most recent
                streak = 0
                for w in past["win"].values[::-1]:
                    if w == past["win"].values[-1]:
                        streak += 1
                    else:
                        break
                features[f"{prefix}_streak"] = streak if past["win"].values[-1] == 1 else -streak

                # Last 5 games form
                last5 = past.tail(5)
                features[f"{prefix}_form_5"] = last5["win"].mean()

                # Home/away specific win rate
                home_past = tlog[(tlog["date"] < game_date) & (tlog["is_home"] == (1 if prefix == "home" else 0))].tail(W)
                if len(home_past) >= 3:
                    features[f"{prefix}_venue_win_pct"] = home_past["win"].mean()
                else:
                    features[f"{prefix}_venue_win_pct"] = features[f"{prefix}_win_pct"]

                # === CONSISTENCY FEATURES (for parlay optimization) ===
                # Point differential consistency (lower std = more predictable)
                pt_diff = past["pts_for"] - past["pts_against"]
                features[f"{prefix}_consistency"] = pt_diff.std() if len(past) >= 5 else 12.0

                # Scoring consistency
                features[f"{prefix}_pts_std"] = past["pts_for"].std() if len(past) >= 5 else 10.0

                # Last 20 games for upset rate (need more data)
                past20 = tlog[tlog["date"] < game_date].tail(20)
                if len(past20) >= 10:
                    features[f"{prefix}_form_20"] = past20["win"].mean()
                else:
                    features[f"{prefix}_form_20"] = features[f"{prefix}_win_pct"]
            else:
                # Not enough data — use defaults
                features[f"{prefix}_win_pct"] = 0.5
                features[f"{prefix}_avg_pts"] = 110
                features[f"{prefix}_avg_pts_allowed"] = 110
                features[f"{prefix}_avg_fg_pct"] = 0.45
                features[f"{prefix}_avg_ft_pct"] = 0.77
                features[f"{prefix}_avg_fg3_pct"] = 0.35
                features[f"{prefix}_avg_ast"] = 23
                features[f"{prefix}_avg_reb"] = 43
                features[f"{prefix}_net_rating"] = 0
                features[f"{prefix}_streak"] = 0
                features[f"{prefix}_form_5"] = 0.5
                features[f"{prefix}_venue_win_pct"] = 0.5
                features[f"{prefix}_consistency"] = 12.0
                features[f"{prefix}_pts_std"] = 10.0
                features[f"{prefix}_form_20"] = 0.5

        # Differential features (most predictive)
        for stat in ["win_pct", "avg_pts", "avg_pts_allowed", "avg_fg_pct", "avg_fg3_pct",
                     "avg_ast", "avg_reb", "net_rating", "form_5", "form_20", "consistency"]:
            h_val = features.get(f"home_{stat}", 0)
            a_val = features.get(f"away_{stat}", 0)
            features[f"diff_{stat}"] = h_val - a_val

        # Rest days approximation (from game log)
        for prefix, team in [("home", ht), ("away", at)]:
            tlog = team_stats.get(team)
            if tlog is not None:
                prev_games = tlog[tlog["date"] < game_date]
                if len(prev_games) > 0:
                    last_game = prev_games["date"].iloc[-1]
                    features[f"{prefix}_rest_days"] = (game_date - last_game).days
                else:
                    features[f"{prefix}_rest_days"] = 3
            else:
                features[f"{prefix}_rest_days"] = 3

        features["rest_diff"] = features.get("home_rest_days", 3) - features.get("away_rest_days", 3)

        # Total points features (for over/under model)
        features["combined_avg_pts"] = features.get("home_avg_pts", 110) + features.get("away_avg_pts", 110)
        features["combined_avg_allowed"] = features.get("home_avg_pts_allowed", 110) + features.get("away_avg_pts_allowed", 110)

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    # Add target columns from original df
    features_df["home_win"] = df["home_win"].values
    features_df["home_pts"] = df["home_pts"].values
    features_df["away_pts"] = df["away_pts"].values
    features_df["total_pts"] = df["home_pts"].values + df["away_pts"].values
    features_df["point_diff"] = df["home_pts"].values - df["away_pts"].values
    features_df["home_team"] = df["home_team"].values
    features_df["away_team"] = df["away_team"].values
    features_df["date"] = df["date"].values

    # Drop rows with too many NaNs (early season games)
    features_df = features_df.dropna(thresh=15).reset_index(drop=True)

    print(f"✅ Built {len(features_df.columns)} features for {len(features_df)} games")
    return features_df


def get_team_current_stats(features_df):
    """Get the most recent rolling stats for each team (for prediction)"""
    latest = features_df.sort_values("date")
    team_stats = {}

    all_teams = set(latest["home_team"].dropna().unique()) | set(latest["away_team"].dropna().unique())

    for team in all_teams:
        # Get most recent game where this team played
        home_rows = latest[latest["home_team"] == team].tail(1)
        away_rows = latest[latest["away_team"] == team].tail(1)

        stats = {}
        if len(home_rows) > 0:
            row = home_rows.iloc[0]
            for col in row.index:
                if col.startswith("home_") and col != "home_team" and col != "home_win" and col != "home_pts":
                    stat_name = col.replace("home_", "")
                    stats[stat_name] = row[col]

        if len(away_rows) > 0:
            row = away_rows.iloc[0]
            for col in row.index:
                if col.startswith("away_") and col != "away_team" and col != "away_pts":
                    stat_name = col.replace("away_", "")
                    # Average with home stats if we have both
                    if stat_name in stats:
                        stats[stat_name] = (stats[stat_name] + row[col]) / 2
                    else:
                        stats[stat_name] = row[col]

        team_stats[team] = stats

    return team_stats


if __name__ == "__main__":
    # Test the pipeline
    df = load_historical_data()
    features = build_features(df)
    print(features.head())
    print(f"\nFeature columns: {[c for c in features.columns if c not in ['home_win', 'home_pts', 'away_pts', 'total_pts', 'point_diff', 'home_team', 'away_team', 'date']]}")
