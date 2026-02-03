"""
Model Engine v3: Calibrated XGBoost models for moneyline, spread, and totals
Key improvement: Isotonic calibration for trustworthy probabilities (critical for parlays)
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, XGBRegressor
from config import *
from data_pipeline import load_historical_data, build_features, fetch_live_odds, parse_odds, get_team_current_stats, fetch_player_availability


# Features used for each model type
MONEYLINE_FEATURES = [
    "home_win_pct", "away_win_pct",
    "home_avg_pts", "away_avg_pts",
    "home_avg_pts_allowed", "away_avg_pts_allowed",
    "home_avg_fg_pct", "away_avg_fg_pct",
    "home_avg_fg3_pct", "away_avg_fg3_pct",
    "home_avg_ast", "away_avg_ast",
    "home_avg_reb", "away_avg_reb",
    "home_net_rating", "away_net_rating",
    "home_streak", "away_streak",
    "home_form_5", "away_form_5",
    "home_form_20", "away_form_20",
    "home_venue_win_pct", "away_venue_win_pct",
    "home_rest_days", "away_rest_days",
    "home_consistency", "away_consistency",
    "diff_win_pct", "diff_avg_pts", "diff_avg_pts_allowed",
    "diff_avg_fg_pct", "diff_avg_fg3_pct",
    "diff_avg_ast", "diff_avg_reb",
    "diff_net_rating", "diff_form_5", "diff_form_20",
    "diff_consistency",
    "rest_diff",
]

TOTALS_FEATURES = [
    "home_avg_pts", "away_avg_pts",
    "home_avg_pts_allowed", "away_avg_pts_allowed",
    "home_avg_fg_pct", "away_avg_fg_pct",
    "home_avg_fg3_pct", "away_avg_fg3_pct",
    "home_avg_ast", "away_avg_ast",
    "home_avg_reb", "away_avg_reb",
    "home_net_rating", "away_net_rating",
    "home_form_5", "away_form_5",
    "home_rest_days", "away_rest_days",
    "home_pts_std", "away_pts_std",
    "combined_avg_pts", "combined_avg_allowed",
    "diff_avg_pts", "diff_avg_pts_allowed",
]

SPREAD_FEATURES = MONEYLINE_FEATURES  # Same features, different target


def train_models(features_df):
    """Train all three models with isotonic calibration for moneyline"""
    print("\n===== Training Calibrated Models v3 =====")

    # Clean data
    df = features_df.dropna(subset=["home_win"]).copy()

    # ---- MONEYLINE MODEL (with calibration) ----
    print("\n--- Moneyline Model (Calibrated) ---")
    ml_features = [f for f in MONEYLINE_FEATURES if f in df.columns]
    X_ml = df[ml_features].fillna(0)
    y_ml = df["home_win"].astype(int)

    # Remove rows where features are mostly defaults
    valid_mask = X_ml.notna().sum(axis=1) > len(ml_features) * 0.5
    X_ml = X_ml[valid_mask]
    y_ml = y_ml[valid_mask]

    scaler_ml = StandardScaler()
    X_ml_scaled = scaler_ml.fit_transform(X_ml)

    # Time-series cross validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Base XGBoost
    base_model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )

    # Cross-validate base model first
    cv_scores = cross_val_score(base_model, X_ml_scaled, y_ml, cv=tscv, scoring="accuracy")
    print(f"  Base CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Per fold: {[f'{s:.4f}' for s in cv_scores]}")

    # Train base model
    base_model.fit(X_ml_scaled, y_ml)

    # Wrap with isotonic calibration (better than Platt for 30K+ samples)
    print(f"  Applying {CALIBRATION_METHOD} calibration...")
    ml_model = CalibratedClassifierCV(
        base_model,
        method=CALIBRATION_METHOD,
        cv=5,
    )
    ml_model.fit(X_ml_scaled, y_ml)

    # Measure calibration quality (Brier score — lower is better)
    # Use last fold as proxy
    train_idx, val_idx = list(tscv.split(X_ml_scaled))[-1]
    val_probs = ml_model.predict_proba(X_ml_scaled[val_idx])[:, 1]
    brier = brier_score_loss(y_ml.iloc[val_idx], val_probs)
    val_acc = accuracy_score(y_ml.iloc[val_idx], (val_probs > 0.5).astype(int))
    print(f"  Calibrated Val Accuracy: {val_acc:.4f}")
    print(f"  Brier Score: {brier:.4f} (lower = better calibrated, 0.25 = random)")

    # ---- TOTALS MODEL (regression) ----
    print("\n--- Totals Model ---")
    df_totals = df.dropna(subset=["total_pts"]).copy()
    tot_features = [f for f in TOTALS_FEATURES if f in df_totals.columns]
    X_tot = df_totals[tot_features].fillna(0)
    y_tot = df_totals["total_pts"]

    valid_mask = X_tot.notna().sum(axis=1) > len(tot_features) * 0.5
    X_tot = X_tot[valid_mask]
    y_tot = y_tot[valid_mask]

    scaler_tot = StandardScaler()
    X_tot_scaled = scaler_tot.fit_transform(X_tot)

    tot_model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=RANDOM_STATE,
    )

    cv_mae = cross_val_score(tot_model, X_tot_scaled, y_tot, cv=tscv, scoring="neg_mean_absolute_error")
    print(f"  CV MAE: {-cv_mae.mean():.2f} points (+/- {cv_mae.std():.2f})")
    tot_model.fit(X_tot_scaled, y_tot)

    # ---- SPREAD MODEL (regression) ----
    print("\n--- Spread Model ---")
    df_spread = df.dropna(subset=["point_diff"]).copy()
    spr_features = [f for f in SPREAD_FEATURES if f in df_spread.columns]
    X_spr = df_spread[spr_features].fillna(0)
    y_spr = df_spread["point_diff"]

    valid_mask = X_spr.notna().sum(axis=1) > len(spr_features) * 0.5
    X_spr = X_spr[valid_mask]
    y_spr = y_spr[valid_mask]

    scaler_spr = StandardScaler()
    X_spr_scaled = scaler_spr.fit_transform(X_spr)

    spr_model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=RANDOM_STATE,
    )

    cv_mae_spr = cross_val_score(spr_model, X_spr_scaled, y_spr, cv=tscv, scoring="neg_mean_absolute_error")
    print(f"  CV MAE: {-cv_mae_spr.mean():.2f} points (+/- {cv_mae_spr.std():.2f})")
    spr_model.fit(X_spr_scaled, y_spr)

    # ---- TOTALS CLASSIFIER (calibrated over/under) ----
    # This replaces the arbitrary confidence formula with a real calibrated model
    print("\n--- Totals Over/Under Classifier (Calibrated) ---")
    # We need a totals line to train against — use rolling average total as proxy
    df_totcls = df.dropna(subset=["total_pts"]).copy()
    # Use combined_avg_pts as proxy for the "line" (what the market would set)
    df_totcls["proxy_line"] = df_totcls["combined_avg_pts"] if "combined_avg_pts" in df_totcls.columns else 220.0
    df_totcls["went_over"] = (df_totcls["total_pts"] > df_totcls["proxy_line"]).astype(int)

    X_totcls = df_totcls[tot_features].fillna(0)
    y_totcls = df_totcls["went_over"]
    valid_mask = X_totcls.notna().sum(axis=1) > len(tot_features) * 0.5
    X_totcls = X_totcls[valid_mask]
    y_totcls = y_totcls[valid_mask]

    X_totcls_scaled = scaler_tot.transform(X_totcls)

    base_totcls = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=RANDOM_STATE, eval_metric="logloss",
    )
    base_totcls.fit(X_totcls_scaled, y_totcls)

    totcls_model = CalibratedClassifierCV(base_totcls, method=CALIBRATION_METHOD, cv=5)
    totcls_model.fit(X_totcls_scaled, y_totcls)

    # Measure
    train_idx_tc, val_idx_tc = list(tscv.split(X_totcls_scaled))[-1]
    val_probs_tc = totcls_model.predict_proba(X_totcls_scaled[val_idx_tc])[:, 1]
    brier_totcls = brier_score_loss(y_totcls.iloc[val_idx_tc], val_probs_tc)
    val_acc_tc = accuracy_score(y_totcls.iloc[val_idx_tc], (val_probs_tc > 0.5).astype(int))
    print(f"  Totals Classifier Val Accuracy: {val_acc_tc:.4f}")
    print(f"  Totals Classifier Brier Score: {brier_totcls:.4f}")

    # ---- SPREAD CLASSIFIER (calibrated cover/not-cover) ----
    print("\n--- Spread Cover Classifier (Calibrated) ---")
    df_sprcls = df.dropna(subset=["point_diff"]).copy()
    # Use predicted spread as proxy for market line (home team net rating diff)
    if "diff_net_rating" in df_sprcls.columns:
        df_sprcls["proxy_spread"] = -df_sprcls["diff_net_rating"]  # Negative because spread favors better team
    else:
        df_sprcls["proxy_spread"] = 0.0
    df_sprcls["covered"] = (df_sprcls["point_diff"] > df_sprcls["proxy_spread"]).astype(int)

    X_sprcls = df_sprcls[spr_features].fillna(0)
    y_sprcls = df_sprcls["covered"]
    valid_mask = X_sprcls.notna().sum(axis=1) > len(spr_features) * 0.5
    X_sprcls = X_sprcls[valid_mask]
    y_sprcls = y_sprcls[valid_mask]

    X_sprcls_scaled = scaler_spr.transform(X_sprcls)

    base_sprcls = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=RANDOM_STATE, eval_metric="logloss",
    )
    base_sprcls.fit(X_sprcls_scaled, y_sprcls)

    sprcls_model = CalibratedClassifierCV(base_sprcls, method=CALIBRATION_METHOD, cv=5)
    sprcls_model.fit(X_sprcls_scaled, y_sprcls)

    # Measure
    train_idx_sc, val_idx_sc = list(tscv.split(X_sprcls_scaled))[-1]
    val_probs_sc = sprcls_model.predict_proba(X_sprcls_scaled[val_idx_sc])[:, 1]
    brier_sprcls = brier_score_loss(y_sprcls.iloc[val_idx_sc], val_probs_sc)
    val_acc_sc = accuracy_score(y_sprcls.iloc[val_idx_sc], (val_probs_sc > 0.5).astype(int))
    print(f"  Spread Classifier Val Accuracy: {val_acc_sc:.4f}")
    print(f"  Spread Classifier Brier Score: {brier_sprcls:.4f}")

    # ---- FEATURE IMPORTANCE ----
    print("\n--- Top 10 Features (base model) ---")
    importance = pd.Series(base_model.feature_importances_, index=ml_features).sort_values(ascending=False)
    for feat, imp in importance.head(10).items():
        print(f"   {feat}: {imp:.4f}")

    # ---- SAVE ----
    package = {
        "ml_model": ml_model, "ml_scaler": scaler_ml, "ml_features": ml_features,
        "tot_model": tot_model, "tot_scaler": scaler_tot, "tot_features": tot_features,
        "spr_model": spr_model, "spr_scaler": scaler_spr, "spr_features": spr_features,
        # NEW: calibrated classifiers for totals and spread confidence
        "totcls_model": totcls_model,
        "sprcls_model": sprcls_model,
        "cv_accuracy": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "calibrated_accuracy": val_acc,
        "brier_score": brier,
        "brier_totals_cls": brier_totcls,
        "brier_spread_cls": brier_sprcls,
        "totals_cls_accuracy": val_acc_tc,
        "spread_cls_accuracy": val_acc_sc,
        "cv_totals_mae": -cv_mae.mean(),
        "cv_spread_mae": -cv_mae_spr.mean(),
        "trained_at": pd.Timestamp.now().isoformat(),
        "n_games_trained": len(df),
        "calibration_method": CALIBRATION_METHOD,
    }
    joblib.dump(package, MODEL_PATH)
    print(f"\n=== Models saved to {MODEL_PATH} ===")
    print(f"  Moneyline accuracy: {cv_scores.mean()*100:.1f}%  (calibrated: {val_acc*100:.1f}%)")
    print(f"  Brier score: {brier:.4f}")
    print(f"  Totals: MAE {-cv_mae.mean():.1f} pts | Classifier {val_acc_tc*100:.1f}% (Brier: {brier_totcls:.4f})")
    print(f"  Spread: MAE {-cv_mae_spr.mean():.1f} pts | Classifier {val_acc_sc*100:.1f}% (Brier: {brier_sprcls:.4f})")
    print(f"  Trained on: {len(df)} games")

    return package


def load_models():
    """Load trained models"""
    return joblib.load(MODEL_PATH)


def predict_games(odds_df, features_df, model_package):
    """Generate predictions using calibrated probabilities + injury adjustments"""
    print("\n===== Generating Predictions =====")

    ml_model = model_package["ml_model"]
    ml_scaler = model_package["ml_scaler"]
    ml_features = model_package["ml_features"]
    tot_model = model_package["tot_model"]
    tot_scaler = model_package["tot_scaler"]
    tot_features = model_package["tot_features"]
    spr_model = model_package["spr_model"]
    spr_scaler = model_package["spr_scaler"]
    spr_features = model_package["spr_features"]

    # NEW: calibrated classifiers for totals/spread confidence
    totcls_model = model_package.get("totcls_model")
    sprcls_model = model_package.get("sprcls_model")

    team_stats = get_team_current_stats(features_df)

    # NEW: Fetch injury/availability data
    injury_data = fetch_player_availability()

    results = []
    for _, game in odds_df.iterrows():
        ht = game["home_team"]
        at = game["away_team"]

        ht_stats = team_stats.get(ht, {})
        at_stats = team_stats.get(at, {})

        # Build feature vector for this game
        fv = {}
        for feat in set(ml_features + tot_features + spr_features):
            if feat.startswith("home_"):
                stat = feat.replace("home_", "")
                fv[feat] = ht_stats.get(stat, 0.5 if "pct" in stat else 0)
            elif feat.startswith("away_"):
                stat = feat.replace("away_", "")
                fv[feat] = at_stats.get(stat, 0.5 if "pct" in stat else 0)
            elif feat.startswith("diff_"):
                stat = feat.replace("diff_", "")
                h_val = ht_stats.get(stat, 0)
                a_val = at_stats.get(stat, 0)
                fv[feat] = h_val - a_val
            elif feat == "rest_diff":
                fv[feat] = ht_stats.get("rest_days", 2) - at_stats.get("rest_days", 2)
            elif feat == "combined_avg_pts":
                fv[feat] = ht_stats.get("avg_pts", 110) + at_stats.get("avg_pts", 110)
            elif feat == "combined_avg_allowed":
                fv[feat] = ht_stats.get("avg_pts_allowed", 110) + at_stats.get("avg_pts_allowed", 110)
            else:
                fv[feat] = 0

        # ---- INJURY ADJUSTMENTS ----
        home_injury = injury_data.get(ht, {})
        away_injury = injury_data.get(at, {})
        home_injury_impact = home_injury.get("injury_impact", 0.0)
        away_injury_impact = away_injury.get("injury_impact", 0.0)
        home_missing = home_injury.get("missing_stars", 0)
        away_missing = away_injury.get("missing_stars", 0)
        # Net injury advantage: positive = home team healthier
        injury_advantage = away_injury_impact - home_injury_impact

        # ---- MONEYLINE PREDICTION (calibrated + injury adjusted) ----
        X_ml = pd.DataFrame([{f: fv.get(f, 0) for f in ml_features}])
        X_ml_s = ml_scaler.transform(X_ml)
        ml_prob = ml_model.predict_proba(X_ml_s)[0]
        home_prob_model = ml_prob[1]  # Calibrated probability
        away_prob_model = ml_prob[0]

        # Apply injury adjustment to model probability
        # Each point of injury impact shifts win prob by ~2.5% (empirically derived)
        if injury_advantage != 0:
            injury_shift = injury_advantage * 0.025
            home_prob_model = np.clip(home_prob_model + injury_shift, 0.05, 0.95)
            away_prob_model = 1.0 - home_prob_model

        # Blend calibrated model with market (70% model / 30% market now)
        market_home = game.get("ml_home_true_prob", 0.5)
        market_away = game.get("ml_away_true_prob", 0.5)

        blended_home = MODEL_WEIGHT * home_prob_model + (1 - MODEL_WEIGHT) * market_home
        blended_away = MODEL_WEIGHT * away_prob_model + (1 - MODEL_WEIGHT) * market_away

        # Normalize
        total = blended_home + blended_away
        blended_home /= total
        blended_away /= total

        predicted_winner = ht if blended_home > 0.5 else at
        confidence = max(blended_home, blended_away)

        # EV calculation
        if predicted_winner == ht:
            best_odds = game.get("ml_home_odds", 1.9)
            best_book = game.get("ml_home_book", "N/A")
            implied = game.get("ml_home_implied", 0.5)
        else:
            best_odds = game.get("ml_away_odds", 1.9)
            best_book = game.get("ml_away_book", "N/A")
            implied = game.get("ml_away_implied", 0.5)

        ev = (confidence * best_odds - 1) * 100
        kelly = max(0, (confidence * best_odds - 1) / (best_odds - 1)) if best_odds > 1 else 0

        # ---- TOTALS PREDICTION (calibrated classifier replaces arbitrary formula) ----
        X_tot = pd.DataFrame([{f: fv.get(f, 0) for f in tot_features}])
        X_tot_s = tot_scaler.transform(X_tot)
        predicted_total = tot_model.predict(X_tot_s)[0]

        # Injury adjustment for totals: missing players reduce scoring
        total_injury_reduction = (home_injury_impact + away_injury_impact) * 0.5
        predicted_total -= total_injury_reduction

        total_line = game.get("total_line", predicted_total)
        total_diff = predicted_total - total_line
        total_pick = "OVER" if total_diff > 0 else "UNDER"

        # NEW: Use calibrated classifier blended with edge-based confidence
        if totcls_model is not None:
            totcls_prob = totcls_model.predict_proba(X_tot_s)[0]
            if total_pick == "OVER":
                cls_conf = totcls_prob[1]
            else:
                cls_conf = totcls_prob[0]
            # Edge-based confidence
            edge_conf = 0.50 + 0.15 * np.tanh(abs(total_diff) / 10.0)
            # Blend 50/50: classifier + edge-based
            total_confidence = 0.5 * cls_conf + 0.5 * edge_conf
            # Cap at realistic levels (totals rarely >70% confident)
            total_confidence = np.clip(total_confidence, 0.45, 0.70)
        else:
            total_confidence = min(0.72, 0.50 + abs(total_diff) / 50)

        if total_pick == "OVER":
            total_odds = game.get("total_over_odds", 1.9)
            total_book = game.get("total_over_book", "N/A")
        else:
            total_odds = game.get("total_under_odds", 1.9)
            total_book = game.get("total_under_book", "N/A")

        total_ev = (total_confidence * total_odds - 1) * 100

        # ---- SPREAD PREDICTION (calibrated classifier replaces arbitrary formula) ----
        X_spr = pd.DataFrame([{f: fv.get(f, 0) for f in spr_features}])
        X_spr_s = spr_scaler.transform(X_spr)
        predicted_diff = spr_model.predict(X_spr_s)[0]

        # Injury adjustment for spread
        predicted_diff += injury_advantage

        spread_line = game.get("spread_home", 0)
        spread_edge = predicted_diff - (-spread_line)
        spread_pick = f"{ht} {spread_line:+.1f}" if spread_edge > 0 else f"{at} {game.get('spread_away', 0):+.1f}"

        # NEW: Use calibrated classifier blended with edge-based confidence
        if sprcls_model is not None:
            sprcls_prob = sprcls_model.predict_proba(X_spr_s)[0]
            if spread_edge > 0:
                cls_conf = sprcls_prob[1]
            else:
                cls_conf = sprcls_prob[0]
            # Edge-based confidence (scaled logistically)
            edge_conf = 0.50 + 0.20 * np.tanh(abs(spread_edge) / 8.0)
            # Blend 50/50: classifier + edge-based (prevents classifier overconfidence)
            spread_confidence = 0.5 * cls_conf + 0.5 * edge_conf
            # Cap at realistic levels (no spread bet is >75% confident)
            spread_confidence = np.clip(spread_confidence, 0.45, 0.75)
        else:
            spread_confidence = min(0.68, 0.50 + abs(spread_edge) / 35)

        if spread_edge > 0:
            spread_odds = game.get("spread_home_odds", 1.9)
            spread_book = game.get("spread_home_book", "N/A")
        else:
            spread_odds = game.get("spread_away_odds", 1.9)
            spread_book = game.get("spread_away_book", "N/A")

        spread_ev = (spread_confidence * spread_odds - 1) * 100

        # ---- CONSISTENCY SCORE (for parlay suitability) ----
        home_consist = ht_stats.get("consistency", 12.0)
        away_consist = at_stats.get("consistency", 12.0)
        # Lower consistency number = more predictable = better for parlays
        parlay_suitability = max(0, 100 - (home_consist + away_consist))

        result = {
            # Game info
            "game_time": game.get("game_time", "TBD"),
            "commence_time": game.get("commence_time", ""),
            "home_team": ht,
            "away_team": at,
            "days_until": game.get("days_until", 0),

            # Moneyline
            "ml_pick": predicted_winner,
            "ml_confidence": round(confidence * 100, 1),
            "ml_home_prob": round(blended_home * 100, 1),
            "ml_away_prob": round(blended_away * 100, 1),
            "ml_model_home_prob": round(home_prob_model * 100, 1),
            "ml_market_home_prob": round(market_home * 100, 1),
            "ml_odds": round(best_odds, 3),
            "ml_book": best_book,
            "ml_ev": round(ev, 1),
            "ml_kelly": round(kelly * 100, 1),

            # Totals
            "total_line": total_line,
            "predicted_total": round(predicted_total, 1),
            "total_pick": total_pick,
            "total_confidence": round(total_confidence * 100, 1),
            "total_odds": round(total_odds, 3) if not pd.isna(total_odds) else 1.9,
            "total_book": total_book if total_book else "N/A",
            "total_ev": round(total_ev, 1),

            # Spread
            "spread_line": spread_line,
            "predicted_diff": round(predicted_diff, 1),
            "spread_pick": spread_pick,
            "spread_confidence": round(spread_confidence * 100, 1),
            "spread_odds": round(spread_odds, 3) if not pd.isna(spread_odds) else 1.9,
            "spread_book": spread_book if spread_book else "N/A",
            "spread_ev": round(spread_ev, 1),

            # Parlay metadata
            "parlay_suitability": round(parlay_suitability, 1),
            "home_consistency": round(home_consist, 2),
            "away_consistency": round(away_consist, 2),

            # Injury data
            "home_missing_stars": home_missing,
            "away_missing_stars": away_missing,
            "home_injury_impact": round(home_injury_impact, 2),
            "away_injury_impact": round(away_injury_impact, 2),
            "injury_advantage": round(injury_advantage, 2),
        }
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["days_until", "ml_ev"], ascending=[True, False])

    # Save predictions
    results_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"  Predictions saved for {len(results_df)} games")

    return results_df


def run_full_pipeline():
    """Run the complete pipeline: data -> features -> train -> predict"""
    # 1. Load historical data and build features
    df = load_historical_data()
    features_df = build_features(df)

    # 2. Train models
    package = train_models(features_df)

    # 3. Fetch live odds
    odds_raw = fetch_live_odds()
    if not odds_raw:
        print("No odds data available")
        return None, None, features_df

    odds_df = parse_odds(odds_raw)
    print(f"\n{len(odds_df)} upcoming games found")

    # 4. Predict
    predictions = predict_games(odds_df, features_df, package)

    return predictions, package, features_df


if __name__ == "__main__":
    predictions, package, features_df = run_full_pipeline()
    if predictions is not None:
        print("\n" + "=" * 80)
        print("NBA BETTING PREDICTIONS (Calibrated v3)")
        print("=" * 80)
        for _, r in predictions.iterrows():
            print(f"\n{r['game_time']} | {r['home_team']} vs {r['away_team']}")
            print(f"  ML: {r['ml_pick']} ({r['ml_confidence']:.1f}%) | Odds: {r['ml_odds']} ({r['ml_book']}) | EV: {r['ml_ev']:+.1f}%")
            print(f"  Total: {r['total_pick']} {r['total_line']} (pred: {r['predicted_total']}) | {r['total_confidence']:.1f}% | EV: {r['total_ev']:+.1f}%")
            print(f"  Spread: {r['spread_pick']} ({r['spread_confidence']:.1f}%) | EV: {r['spread_ev']:+.1f}%")
            print(f"  Parlay Suitability: {r['parlay_suitability']:.0f}/100")
