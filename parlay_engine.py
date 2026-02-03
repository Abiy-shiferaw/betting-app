"""
Parlay Engine v3.1: Data-driven parlay builder with measured correlations
Core philosophy: HIGH WIN RATE picks that actually hit, combined smartly
Key improvement: Real correlation values from 30K+ game analysis replace guesswork
"""
import pandas as pd
import numpy as np
from itertools import combinations
from config import *


def get_bet_type_key(bet_type):
    """Normalize bet type string to key"""
    bt = bet_type.lower()
    if "money" in bt or bt == "moneyline":
        return "ml"
    elif "over" in bt or "under" in bt or "total" in bt:
        return "total"
    elif "spread" in bt:
        return "spread"
    return "ml"


def get_correlation(leg1, leg2, same_game=False):
    """
    Get DATA-DRIVEN correlation between two parlay legs.
    Uses real measured correlations from 30,336 historical NBA games.

    Returns: correlation factor (0.0=independent, 1.0=fully correlated)
    """
    bt1 = get_bet_type_key(leg1.get("bet_type", "Moneyline"))
    bt2 = get_bet_type_key(leg2.get("bet_type", "Moneyline"))

    # Sort for consistent lookup
    pair = tuple(sorted([bt1, bt2]))

    if same_game:
        # Same-game parlay correlations (measured from real data)
        if pair == ("ml", "spread"):
            return CORRELATION_MATRIX["ml_spread_same_game"]   # 0.80 — DANGER
        elif pair == ("ml", "total"):
            return CORRELATION_MATRIX["ml_total_same_game"]    # 0.02 — safe
        elif pair == ("spread", "total"):
            return CORRELATION_MATRIX["spread_total_same_game"]  # 0.02 — safe
        elif pair == ("ml", "ml") or pair == ("spread", "spread") or pair == ("total", "total"):
            return 1.0  # Can't bet same type twice on same game
        return 0.10  # Unknown same-game combo
    else:
        # Cross-game correlations (all nearly independent)
        if pair == ("ml", "ml"):
            return CORRELATION_MATRIX["ml_ml_cross_game"]      # 0.007
        elif pair == ("total", "total"):
            return CORRELATION_MATRIX["total_total_cross_game"]  # 0.11
        elif pair == ("spread", "spread"):
            return CORRELATION_MATRIX["spread_spread_cross_game"]  # 0.01
        elif pair == ("ml", "spread"):
            return CORRELATION_MATRIX["ml_spread_cross_game"]   # 0.01
        elif pair == ("ml", "total"):
            return CORRELATION_MATRIX["ml_total_cross_game"]    # 0.01
        elif pair == ("spread", "total"):
            return CORRELATION_MATRIX["spread_total_cross_game"]  # 0.01
        return CORRELATION_MATRIX.get("division_bonus", 0.001)  # Default


def calculate_parlay_probability(legs, correlation_matrix=None):
    """
    Calculate true parlay win probability.
    Multiplies individual probabilities, applies correlation adjustment.
    """
    if not legs:
        return 0.0

    # Base: multiply independent probabilities
    prob = 1.0
    for leg in legs:
        prob *= leg["confidence"] / 100.0

    # Apply correlation penalty if significant
    if correlation_matrix:
        avg_corr = np.mean(list(correlation_matrix.values())) if correlation_matrix else 0
        # Correlation reduces independence — slightly lower combined probability
        prob *= (1.0 - avg_corr * 0.1)

    return prob


def calculate_parlay_odds(legs):
    """
    Calculate combined parlay odds (decimal).
    Multiply individual decimal odds.
    """
    combined = 1.0
    for leg in legs:
        combined *= leg["odds"]
    return combined


def calculate_parlay_ev(win_prob, combined_odds):
    """
    Calculate parlay expected value.
    EV = (probability * payout) - 1
    """
    return (win_prob * combined_odds - 1) * 100


def calculate_kelly(probability, odds, fraction=None):
    """
    Kelly criterion for optimal bet sizing.
    Uses fractional Kelly for safety.
    """
    if fraction is None:
        fraction = PARLAY_KELLY_FRACTION
    if odds <= 1:
        return 0
    full_kelly = (probability * odds - 1) / (odds - 1)
    return max(0, full_kelly * fraction) * 100  # Return as percentage


def grade_parlay(win_prob, ev, num_legs):
    """
    Grade a parlay A/B/C+/D based on win probability and EV.
    Parlays need different thresholds than single bets.
    """
    if ev <= 0:
        return "D", "SKIP", "#ff4757"

    if num_legs == 2:
        if win_prob >= 0.42 and ev >= 10:
            return "A", "STRONG PARLAY", "#00d4aa"
        if win_prob >= 0.36 and ev >= 5:
            return "B", "SOLID PARLAY", "#00d4aa"
        if win_prob >= 0.30:
            return "C+", "LEAN PARLAY", "#ffa502"
        return "C", "MARGINAL", "#8b8d97"
    else:  # 3 legs
        if win_prob >= 0.28 and ev >= 15:
            return "A", "STRONG PARLAY", "#00d4aa"
        if win_prob >= 0.22 and ev >= 8:
            return "B", "SOLID PARLAY", "#00d4aa"
        if win_prob >= 0.18:
            return "C+", "LEAN PARLAY", "#ffa502"
        return "C", "MARGINAL", "#8b8d97"


def build_leg_from_prediction(row, bet_type="moneyline"):
    """
    Extract a single parlay leg from a prediction row.
    bet_type: "moneyline", "totals", or "spread"
    """
    base = {
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "game_time": row.get("game_time", "TBD"),
        "commence_time": row.get("commence_time", ""),
    }

    if bet_type == "moneyline":
        return {
            **base,
            "bet_type": "Moneyline",
            "pick": row["ml_pick"],
            "confidence": row["ml_confidence"],
            "odds": row["ml_odds"],
            "ev": row["ml_ev"],
            "book": row.get("ml_book", "N/A"),
            "detail": f"ML {row['ml_pick']}",
            "parlay_suitability": row.get("parlay_suitability", 50),
        }
    elif bet_type == "totals":
        return {
            **base,
            "bet_type": "Over/Under",
            "pick": f"{row['total_pick']} {row.get('total_line', '')}",
            "confidence": row["total_confidence"],
            "odds": row.get("total_odds", 1.9),
            "ev": row["total_ev"],
            "book": row.get("total_book", "N/A"),
            "detail": f"{row['total_pick']} {row.get('total_line', '')} (pred: {row.get('predicted_total', 0):.0f})",
            "parlay_suitability": row.get("parlay_suitability", 50),
        }
    elif bet_type == "spread":
        return {
            **base,
            "bet_type": "Spread",
            "pick": row["spread_pick"],
            "confidence": row["spread_confidence"],
            "odds": row.get("spread_odds", 1.9),
            "ev": row["spread_ev"],
            "book": row.get("spread_book", "N/A"),
            "detail": f"{row['spread_pick']}",
            "parlay_suitability": row.get("parlay_suitability", 50),
        }

    return None


def build_optimal_parlays(predictions_df, max_legs=None, top_n=10):
    """
    Build optimal parlays from prediction data.

    Algorithm:
    1. Filter to high-confidence, positive-EV legs
    2. Generate all 2-leg and 3-leg combinations
    3. Score each combo by composite: 50% win prob + 30% EV + 20% consistency
    4. Filter by minimum thresholds
    5. Return top N parlays ranked by score

    Returns DataFrame of parlay recommendations.
    """
    if max_legs is None:
        max_legs = PARLAY_MAX_LEGS

    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame()

    # Step 1: Build all eligible legs across all bet types
    all_legs = []

    for idx, row in predictions_df.iterrows():
        # Moneyline legs
        if row["ml_confidence"] >= PARLAY_MIN_CONFIDENCE and row["ml_ev"] >= PARLAY_MIN_EV:
            leg = build_leg_from_prediction(row, "moneyline")
            if leg:
                leg["game_idx"] = idx
                all_legs.append(leg)

        # Totals legs
        if row["total_confidence"] >= PARLAY_MIN_CONFIDENCE and row["total_ev"] >= PARLAY_MIN_EV:
            leg = build_leg_from_prediction(row, "totals")
            if leg:
                leg["game_idx"] = idx
                all_legs.append(leg)

        # Spread legs
        if row["spread_confidence"] >= PARLAY_MIN_CONFIDENCE and row["spread_ev"] >= PARLAY_MIN_EV:
            leg = build_leg_from_prediction(row, "spread")
            if leg:
                leg["game_idx"] = idx
                all_legs.append(leg)

    if len(all_legs) < 2:
        print(f"  Only {len(all_legs)} eligible legs found (need 2+). Lowering thresholds...")
        # Try with lower thresholds
        for idx, row in predictions_df.iterrows():
            if row["ml_confidence"] >= 55 and row["ml_ev"] >= -2:
                leg = build_leg_from_prediction(row, "moneyline")
                if leg and not any(l["game_idx"] == idx and l["bet_type"] == "Moneyline" for l in all_legs):
                    leg["game_idx"] = idx
                    all_legs.append(leg)

    print(f"  Found {len(all_legs)} eligible parlay legs")

    if len(all_legs) < 2:
        return pd.DataFrame()

    # Step 2: Generate combinations (2-leg and optionally 3-leg)
    parlays = []

    # 2-leg parlays
    for combo in combinations(range(len(all_legs)), 2):
        legs = [all_legs[i] for i in combo]

        game_idxs = [l["game_idx"] for l in legs]
        is_sgp = len(set(game_idxs)) < len(game_idxs)

        if not is_sgp:
            # Cross-game: check no team in both games
            team_sets = [{l["home_team"], l["away_team"]} for l in legs]
            if team_sets[0] & team_sets[1]:
                continue

        # Calculate DATA-DRIVEN correlation
        corr_pairs = {}
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                corr = get_correlation(legs[i], legs[j], same_game=is_sgp)
                corr_pairs[f"{i}-{j}"] = corr

        # Skip if too correlated (blocks ML+Spread same game at 0.80 > 0.50 threshold)
        max_corr = max(corr_pairs.values()) if corr_pairs else 0
        if max_corr >= CORRELATION_BLOCK_THRESHOLD:
            continue

        # Calculate parlay metrics
        win_prob = calculate_parlay_probability(legs, corr_pairs)
        combined_odds = calculate_parlay_odds(legs)
        ev = calculate_parlay_ev(win_prob, combined_odds)
        kelly = calculate_kelly(win_prob, combined_odds)
        grade, action, color = grade_parlay(win_prob, ev, 2)

        # Composite score: 50% win prob + 30% EV + 20% consistency
        win_score = min(win_prob / 0.50, 1.0) * 50  # Normalize: 50% parlay win = max
        ev_score = min(max(ev, 0) / 30, 1.0) * 30  # Normalize: 30% EV = max
        avg_suitability = np.mean([l.get("parlay_suitability", 50) for l in legs])
        consist_score = (avg_suitability / 100) * 20
        composite = win_score + ev_score + consist_score

        if win_prob >= PARLAY_MIN_WIN_RATE and ev > 0:
            parlays.append({
                "legs": legs,
                "num_legs": 2,
                "win_prob": round(win_prob * 100, 1),
                "combined_odds": round(combined_odds, 2),
                "ev": round(ev, 1),
                "kelly_pct": round(kelly, 1),
                "composite_score": round(composite, 1),
                "grade": grade,
                "action": action,
                "color": color,
                "is_sgp": is_sgp,
                "max_correlation": round(max_corr, 3),
            })

    # 3-leg parlays (if enough legs and max_legs allows)
    if max_legs >= 3 and len(all_legs) >= 3:
        for combo in combinations(range(len(all_legs)), 3):
            legs = [all_legs[i] for i in combo]

            # Check for team overlap across different games
            game_idxs = [l["game_idx"] for l in legs]
            unique_games = set(game_idxs)

            # Skip if more than 2 legs from same game
            from collections import Counter
            game_counts = Counter(game_idxs)
            if max(game_counts.values()) > 2:
                continue

            # Check team overlap for cross-game legs
            skip = False
            for i in range(len(legs)):
                for j in range(i + 1, len(legs)):
                    if game_idxs[i] != game_idxs[j]:
                        teams_i = {legs[i]["home_team"], legs[i]["away_team"]}
                        teams_j = {legs[j]["home_team"], legs[j]["away_team"]}
                        if teams_i & teams_j:
                            skip = True
                            break
                if skip:
                    break
            if skip:
                continue

            # Calculate DATA-DRIVEN correlation
            corr_pairs = {}
            for i in range(len(legs)):
                for j in range(i + 1, len(legs)):
                    same_game = (game_idxs[i] == game_idxs[j])
                    corr = get_correlation(legs[i], legs[j], same_game=same_game)
                    corr_pairs[f"{i}-{j}"] = corr

            max_corr = max(corr_pairs.values()) if corr_pairs else 0
            if max_corr >= CORRELATION_BLOCK_THRESHOLD:
                continue

            win_prob = calculate_parlay_probability(legs, corr_pairs)
            combined_odds = calculate_parlay_odds(legs)
            ev = calculate_parlay_ev(win_prob, combined_odds)
            kelly = calculate_kelly(win_prob, combined_odds)
            grade, action, color = grade_parlay(win_prob, ev, 3)

            win_score = min(win_prob / 0.35, 1.0) * 50
            ev_score = min(max(ev, 0) / 50, 1.0) * 30
            avg_suitability = np.mean([l.get("parlay_suitability", 50) for l in legs])
            consist_score = (avg_suitability / 100) * 20
            composite = win_score + ev_score + consist_score

            if win_prob >= PARLAY_MIN_WIN_RATE * 0.8 and ev > 0:  # Slightly lower threshold for 3-leg
                parlays.append({
                    "legs": legs,
                    "num_legs": 3,
                    "win_prob": round(win_prob * 100, 1),
                    "combined_odds": round(combined_odds, 2),
                    "ev": round(ev, 1),
                    "kelly_pct": round(kelly, 1),
                    "composite_score": round(composite, 1),
                    "grade": grade,
                    "action": action,
                    "color": color,
                    "is_sgp": len(unique_games) < 3,
                    "max_correlation": round(max_corr, 3),
                })

    # Step 3: Rank by composite score and deduplicate
    parlays.sort(key=lambda x: x["composite_score"], reverse=True)

    # Remove very similar parlays (same games, similar legs)
    seen = set()
    unique_parlays = []
    for p in parlays:
        # Create a key from the leg picks
        key = tuple(sorted([(l["home_team"], l["away_team"], l["bet_type"], l["pick"]) for l in p["legs"]]))
        if key not in seen:
            seen.add(key)
            unique_parlays.append(p)

    # Return top N
    result = unique_parlays[:top_n]

    print(f"  Built {len(unique_parlays)} unique parlays, returning top {min(top_n, len(result))}")
    if result:
        best = result[0]
        print(f"  Best parlay: {best['grade']} | {best['num_legs']} legs | Win: {best['win_prob']}% | EV: {best['ev']:+.1f}% | Odds: {best['combined_odds']}")

    return result


def format_parlay_summary(parlay):
    """Format a parlay for display"""
    lines = []
    lines.append(f"{'='*50}")
    lines.append(f"  {parlay['grade']} {parlay['action']} | {parlay['num_legs']} Legs")
    lines.append(f"  Win Rate: {parlay['win_prob']}% | Payout: {parlay['combined_odds']}x | EV: {parlay['ev']:+.1f}%")
    lines.append(f"  Kelly Bet: {parlay['kelly_pct']:.1f}% of bankroll")
    if parlay['is_sgp']:
        lines.append(f"  (Same-Game Parlay)")
    lines.append(f"  {'─'*46}")

    for i, leg in enumerate(parlay["legs"], 1):
        lines.append(f"  Leg {i}: {leg['bet_type']} — {leg['pick']}")
        lines.append(f"         {leg['home_team']} vs {leg['away_team']} ({leg['game_time']})")
        lines.append(f"         Conf: {leg['confidence']:.0f}% | Odds: {leg['odds']} | EV: {leg['ev']:+.1f}%")

    lines.append(f"{'='*50}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with existing predictions
    try:
        predictions = pd.read_csv(PREDICTIONS_PATH)
        print(f"Loaded {len(predictions)} predictions")

        parlays = build_optimal_parlays(predictions)

        print(f"\n{'='*60}")
        print(f" TOP PARLAY RECOMMENDATIONS")
        print(f"{'='*60}")

        for p in parlays:
            print(format_parlay_summary(p))
            print()

    except FileNotFoundError:
        print("No predictions file found. Run model_engine.py first.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
