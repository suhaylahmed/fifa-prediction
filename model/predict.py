"""
Prediction engine — loads artifacts, runs inference, returns structured dict.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from data.preprocess import build_feature_row
from data.world_cup_2026 import (
    adjust_probabilities_for_2026_context,
    get_team_profile,
    get_world_cup_2026_key_players,
)
from model.features import FEATURE_COLUMNS, MODEL_INPUT_COLUMNS, RAW_CONTEXT_COLUMNS, normalize_tournament_name
from utils.supersub import detect_supersub
from utils.player_stats import get_key_players

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")


def _load_artifacts():
    if not all(os.path.exists(p) for p in [MODEL_PATH, FEATURES_PATH]):
        raise FileNotFoundError("Model artifacts not found. Run `python -m model.train` first.")
    artifact = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact, feature_cols

    # Backwards compatibility for older model.pkl artifacts.
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return {
        "model": artifact,
        "model_name": type(artifact).__name__,
        "feature_columns": feature_cols,
        "categorical_columns": [],
        "raw_context_columns": [],
        "numeric_feature_columns": feature_cols,
        "scaler": scaler,
    }, feature_cols


def _get_prob(proba_row: np.ndarray, model, class_label: int) -> float:
    """Safe probability lookup by class label (handles non-contiguous classes_)."""
    idx = np.where(model.classes_ == class_label)[0]
    if len(idx) == 0:
        return 0.0
    return float(proba_row[idx[0]])


def _build_model_input(team_a: str, team_b: str, feat_dict: dict, feature_cols: list[str]) -> pd.DataFrame:
    row = {
        "team_a": team_a,
        "team_b": team_b,
        "tournament": normalize_tournament_name("FIFA World Cup"),
    }
    row.update(feat_dict)

    columns = feature_cols or MODEL_INPUT_COLUMNS
    data = {}
    for col in columns:
        if col in RAW_CONTEXT_COLUMNS:
            data[col] = [str(row.get(col, "unknown") or "unknown")]
        else:
            data[col] = [float(row.get(col, 0.0) or 0.0)]
    return pd.DataFrame(data, columns=columns)


def _positive_probability(model, proba: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    idx = np.where(classes == 1)[0]
    if len(idx) == 0:
        return np.zeros(len(proba), dtype=np.float64)
    return proba[:, int(idx[0])].astype(np.float64)


def _normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _predict_proba(model_bundle: dict, x: pd.DataFrame) -> np.ndarray:
    feature_cols = model_bundle.get("feature_columns") or list(x.columns)
    x_model = x[feature_cols]

    if model_bundle.get("result_model_type") == "two_stage":
        draw_model = model_bundle["draw_model"]
        decisive_model = model_bundle["decisive_model"]
        p_draw = _positive_probability(draw_model, draw_model.predict_proba(x_model))
        p_win_given_decisive = _positive_probability(
            decisive_model,
            decisive_model.predict_proba(x_model),
        )
        p_decisive = 1.0 - p_draw
        return _normalize_probabilities(
            np.column_stack(
                [
                    p_decisive * (1.0 - p_win_given_decisive),
                    p_draw,
                    p_decisive * p_win_given_decisive,
                ]
            )
        )[0]

    model = model_bundle["model"]
    scaler = model_bundle.get("scaler")
    if scaler is not None:
        raw_proba = model.predict_proba(scaler.transform(x_model.to_numpy(dtype=np.float32)))
    else:
        raw_proba = model.predict_proba(x_model)

    probs = np.zeros((len(x_model), 3), dtype=np.float64)
    for idx, class_label in enumerate(model.classes_):
        probs[:, int(class_label)] = raw_proba[:, idx]
    return _normalize_probabilities(probs)[0]


def _confidence_score(
    win_prob: float,
    draw_prob: float,
    loss_prob: float,
    active_groups: int,
    total_groups: int,
    h2h_meetings: int,
) -> tuple[float, str]:
    base = max(win_prob, draw_prob, loss_prob)
    data_coverage = active_groups / total_groups if total_groups > 0 else 0.0
    h2h_bonus = min(h2h_meetings / 20, 0.1)
    raw = (base * 0.7) + (data_coverage * 0.2) + h2h_bonus
    score = float(np.clip(raw, 0.0, 1.0))
    label = (
        "Very High" if score > 0.80 else
        "High" if score > 0.65 else
        "Medium" if score > 0.50 else
        "Low"
    )
    return score, label


def _build_key_factors(team_a: str, team_b: str, feat: dict, ss_a, ss_b) -> list[str]:
    factors = []

    # H2H
    total = int(feat.get("h2h_total_meetings", 0))
    if total > 0:
        wa = int(feat.get("h2h_team_a_wins", 0))
        wb = int(feat.get("h2h_team_b_wins", 0))
        wd = int(feat.get("h2h_draws", 0))
        factors.append(f"{total} WC meetings: {team_a} won {wa}, {team_b} won {wb}, {wd} draws.")
    else:
        factors.append(f"{team_a} and {team_b} have never met at a World Cup.")

    # Form
    wins_a = int(feat.get("team_a_form_wins", 0))
    wins_b = int(feat.get("team_b_form_wins", 0))
    factors.append(
        f"Recent form (last 10): {team_a} — {wins_a}W/{int(feat.get('team_a_form_draws',0))}D/{int(feat.get('team_a_form_losses',0))}L; "
        f"{team_b} — {wins_b}W/{int(feat.get('team_b_form_draws',0))}D/{int(feat.get('team_b_form_losses',0))}L."
    )

    # Rankings
    rank_a = feat.get("team_a_rank", 0)
    rank_b = feat.get("team_b_rank", 0)
    if rank_a and rank_b and rank_a > 0 and rank_b > 0:
        ranking_date = feat.get("display_rank_date")
        date_suffix = f" ({ranking_date})" if ranking_date else ""
        factors.append(
            f"FIFA ranking snapshot{date_suffix}: "
            f"{team_a} #{int(rank_a)} vs {team_b} #{int(rank_b)}."
        )

    # Penalty shootout record (always relevant for WC)
    so_w_a = int(feat.get("team_a_penalty_shootout_wins", 0))
    so_l_a = int(feat.get("team_a_penalty_shootout_losses", 0))
    so_w_b = int(feat.get("team_b_penalty_shootout_wins", 0))
    so_l_b = int(feat.get("team_b_penalty_shootout_losses", 0))
    if so_w_a + so_l_a + so_w_b + so_l_b > 0:
        factors.append(
            f"Penalty shootout record: {team_a} {so_w_a}W/{so_l_a}L; {team_b} {so_w_b}W/{so_l_b}L."
        )

    # Super-sub
    for ss, team in [(ss_a, team_a), (ss_b, team_b)]:
        if ss:
            factors.append(
                f"Super-sub: {ss['name']} ({team}) — "
                f"{ss['goals_as_sub']} goals in {ss['sub_appearances']} sub apps."
            )

    return factors[:5]


def _plain_verdict(
    team_a: str,
    team_b: str,
    win_prob: float,
    draw_prob: float,
    loss_prob: float,
    predicted_label: int,
) -> str:
    if predicted_label == 2:
        margin = win_prob - max(draw_prob, loss_prob)
        strength = "convincingly" if margin > 0.2 else "narrowly"
        return (
            f"The model predicts {team_a} to win {strength} "
            f"({win_prob:.0%} probability). "
            f"A draw is {draw_prob:.0%} likely, and {team_b} winning is {loss_prob:.0%}."
        )
    if predicted_label == 0:
        margin = loss_prob - max(win_prob, draw_prob)
        strength = "convincingly" if margin > 0.2 else "narrowly"
        return (
            f"The model predicts {team_b} to win {strength} "
            f"({loss_prob:.0%} probability). "
            f"A draw is {draw_prob:.0%} likely, and {team_a} winning is {win_prob:.0%}."
        )
    return (
        f"The calibrated draw model identifies this as a close contest "
        f"({draw_prob:.0%} draw probability). {team_a} win probability: {win_prob:.0%}. "
        f"{team_b} win probability: {loss_prob:.0%}."
    )


def _venue_context(venue_mode: str) -> dict[str, float]:
    if venue_mode == "team_a_home":
        return {"team_a_is_home": 1.0, "team_b_is_home": 0.0, "neutral_venue": 0.0}
    if venue_mode == "team_b_home":
        return {"team_a_is_home": 0.0, "team_b_is_home": 1.0, "neutral_venue": 0.0}
    return {"team_a_is_home": 0.0, "team_b_is_home": 0.0, "neutral_venue": 1.0}


def _venue_label(venue_mode: str, team_a: str, team_b: str) -> str:
    if venue_mode == "team_a_home":
        return f"{team_a} home"
    if venue_mode == "team_b_home":
        return f"{team_b} home"
    return "Neutral venue"


def _opposite_venue_mode(venue_mode: str) -> str:
    if venue_mode == "team_a_home":
        return "team_b_home"
    if venue_mode == "team_b_home":
        return "team_a_home"
    return "neutral"


def _prediction_label(model_bundle: dict, probs: np.ndarray) -> tuple[int, str]:
    if model_bundle.get("result_model_type") == "two_stage":
        threshold = float(model_bundle.get("draw_threshold", 0.37))
        if float(probs[1]) >= threshold:
            return 1, f"calibrated draw threshold ({threshold:.0%})"
        return (2 if probs[2] >= probs[0] else 0), "two-stage decisive model"

    best_non_draw = max(float(probs[0]), float(probs[2]))
    threshold = float(model_bundle.get("draw_threshold", 0.24))
    if float(probs[1]) >= threshold and best_non_draw - float(probs[1]) <= 0.15:
        return 1, f"calibrated draw threshold ({threshold:.0%})"
    return int(np.argmax(probs)), "highest model probability"


def predict_match(
    team_a: str,
    team_b: str,
    matches_df: pd.DataFrame,
    goalscorers_df,
    shootouts_df,
    rankings_df,
    substitutions_df=None,
    player_appearances_df=None,
    player_goals_df=None,
    award_winners_df=None,
    copa_data=None,
    euro_data=None,
    friendlies_data=None,
    world_cup_2026_data=None,
    venue_mode: str = "neutral",
) -> dict:
    """
    Run inference for team_a vs team_b.

    Returns a dict with keys:
        team_a, team_b, tournament, stage,
        win_prob, draw_prob, loss_prob,
        confidence_score, confidence_label,
        n_features_used, n_features_total,
        key_factors, verdict,
        supersub_a, supersub_b,
        team_a_rank, team_b_rank,
        team_a_wc_titles, team_b_wc_titles,
        team_a_avg_goals_wc, team_b_avg_goals_wc,
        team_a_clean_sheet_rate, team_b_clean_sheet_rate,
        team_a_penalty_shootout_wins, team_a_penalty_shootout_losses,
        team_b_penalty_shootout_wins, team_b_penalty_shootout_losses,
    """
    model_bundle, feature_cols = _load_artifacts()

    # Use today as reference so predictions use the most current rankings.
    # The leakage guard (date < as_of_date) still works — all historical matches
    # pre-date today, so nothing is excluded; we just get the newest rankings.
    as_of_date = pd.Timestamp.now().normalize()

    TOTAL_GROUPS = 9
    feature_kwargs = {
        "as_of_date": as_of_date,
        "matches_df": matches_df,
        "goalscorers_df": goalscorers_df,
        "shootouts_df": shootouts_df,
        "rankings_df": rankings_df,
        "substitutions_df": substitutions_df,
        "player_appearances_df": player_appearances_df,
        "player_goals_df": player_goals_df,
        "award_winners_df": award_winners_df,
        "copa_data": copa_data,
        "friendlies_data": friendlies_data,
        "euro_data": euro_data,
        "skip_goalscoring": False,
    }
    feat_dict, active_groups = build_feature_row(
        team_a=team_a,
        team_b=team_b,
        venue_context=_venue_context(venue_mode),
        **feature_kwargs,
    )
    mirrored_feat, _ = build_feature_row(
        team_a=team_b,
        team_b=team_a,
        venue_context=_venue_context(_opposite_venue_mode(venue_mode)),
        **feature_kwargs,
    )

    x = _build_model_input(team_a, team_b, feat_dict, feature_cols)
    mirrored_x = _build_model_input(team_b, team_a, mirrored_feat, feature_cols)
    direct_probs = _predict_proba(model_bundle, x)
    mirrored_probs = _predict_proba(model_bundle, mirrored_x)
    proba = _normalize_probabilities(
        np.array(
            [[
                (direct_probs[0] + mirrored_probs[2]) / 2.0,
                (direct_probs[1] + mirrored_probs[1]) / 2.0,
                (direct_probs[2] + mirrored_probs[0]) / 2.0,
            ]]
        )
    )[0]
    loss_prob, draw_prob, win_prob = map(float, proba)
    base_probabilities = {
        "win_prob": win_prob,
        "draw_prob": draw_prob,
        "loss_prob": loss_prob,
    }
    win_prob, draw_prob, loss_prob, world_cup_context = adjust_probabilities_for_2026_context(
        team_a,
        team_b,
        win_prob,
        draw_prob,
        loss_prob,
        world_cup_2026_data,
    )
    world_cup_context.update(
        {
            "team_a_squad_edge_shift": float(world_cup_context.get("win_probability_shift", 0.0)),
            "team_b_squad_edge_shift": -float(world_cup_context.get("win_probability_shift", 0.0)),
            "draw_context_shift": float(world_cup_context.get("draw_probability_shift", 0.0)),
            "team_a_probability_shift": win_prob - base_probabilities["win_prob"],
            "draw_probability_shift": draw_prob - base_probabilities["draw_prob"],
            "team_b_probability_shift": loss_prob - base_probabilities["loss_prob"],
        }
    )

    conf_score, conf_label = _confidence_score(
        win_prob, draw_prob, loss_prob,
        active_groups, TOTAL_GROUPS,
        int(feat_dict.get("h2h_total_meetings", 0)),
    )

    # Super-sub full data for display (reuse filtered slices already computed)
    latest_matches = matches_df[matches_df["date"] < as_of_date]
    latest_goals = (
        goalscorers_df[goalscorers_df["date"] < as_of_date]
        if goalscorers_df is not None else None
    )
    latest_subs = (
        substitutions_df[substitutions_df["date"] < as_of_date]
        if substitutions_df is not None else None
    )
    ss_a = detect_supersub(team_a, latest_goals, latest_matches, latest_subs)
    ss_b = detect_supersub(team_b, latest_goals, latest_matches, latest_subs)

    # Key players for UI display
    kp_a = get_world_cup_2026_key_players(team_a, world_cup_2026_data)
    kp_b = get_world_cup_2026_key_players(team_b, world_cup_2026_data)
    if not kp_a:
        kp_a = get_key_players(team_a, as_of_date, player_appearances_df, player_goals_df, award_winners_df)
    if not kp_b:
        kp_b = get_key_players(team_b, as_of_date, player_appearances_df, player_goals_df, award_winners_df)

    profile_a = get_team_profile(team_a, world_cup_2026_data) or {}
    profile_b = get_team_profile(team_b, world_cup_2026_data) or {}
    model_rank_a = int(feat_dict.get("team_a_rank", 0))
    model_rank_b = int(feat_dict.get("team_b_rank", 0))
    model_rank_date = None
    if rankings_df is not None and "rank_date" in rankings_df.columns and not rankings_df.empty:
        latest_rank_date = pd.to_datetime(rankings_df["rank_date"], errors="coerce").max()
        if pd.notna(latest_rank_date):
            model_rank_date = str(latest_rank_date.date())

    display_feat = dict(feat_dict)
    rank_snapshot_date = (
        world_cup_2026_data.get("ranking_snapshot_date")
        if world_cup_2026_data else None
    )
    if profile_a.get("fifa_ranking", 999) < 999:
        display_feat["team_a_rank"] = profile_a["fifa_ranking"]
        team_a_rank_source = "Official FIFA snapshot"
        team_a_rank_date = rank_snapshot_date
    else:
        team_a_rank_source = "Model ranking snapshot"
        team_a_rank_date = model_rank_date
    if profile_b.get("fifa_ranking", 999) < 999:
        display_feat["team_b_rank"] = profile_b["fifa_ranking"]
        team_b_rank_source = "Official FIFA snapshot"
        team_b_rank_date = rank_snapshot_date
    else:
        team_b_rank_source = "Model ranking snapshot"
        team_b_rank_date = model_rank_date
    if team_a_rank_date == team_b_rank_date:
        display_feat["display_rank_date"] = team_a_rank_date
    key_factors = _build_key_factors(team_a, team_b, display_feat, ss_a, ss_b)

    probabilities = np.array([loss_prob, draw_prob, win_prob], dtype=np.float64)
    predicted_label, decision_policy = _prediction_label(model_bundle, probabilities)
    verdict = _plain_verdict(
        team_a,
        team_b,
        win_prob,
        draw_prob,
        loss_prob,
        predicted_label,
    )

    goal_prediction = {}
    try:
        from model.goals import predict_goals

        goal_prediction = predict_goals(
            team_a=team_a,
            team_b=team_b,
            rankings_df=rankings_df,
            venue_mode=venue_mode,
        )
    except Exception as exc:
        goal_prediction = {"goals_error": str(exc)}

    return {
        "team_a": team_a,
        "team_b": team_b,
        "win_prob": win_prob,
        "draw_prob": draw_prob,
        "loss_prob": loss_prob,
        "confidence_score": round(conf_score, 4),
        "confidence_label": conf_label,
        "n_features_used": active_groups,
        "n_features_total": TOTAL_GROUPS,
        "predicted_label": predicted_label,
        "decision_policy": decision_policy,
        "venue_label": _venue_label(venue_mode, team_a, team_b),
        "base_probabilities": base_probabilities,
        "world_cup_2026_context": world_cup_context,
        "team_a_2026_squad": profile_a.get("official_squad") or {},
        "team_b_2026_squad": profile_b.get("official_squad") or {},
        "key_factors": key_factors,
        "verdict": verdict,
        "supersub_a": ss_a,
        "supersub_b": ss_b,
        "key_players_a": kp_a,
        "key_players_b": kp_b,
        "team_a_rank": int(display_feat.get("team_a_rank", 0)),
        "team_b_rank": int(display_feat.get("team_b_rank", 0)),
        "team_a_rank_source": team_a_rank_source,
        "team_b_rank_source": team_b_rank_source,
        "team_a_rank_date": team_a_rank_date,
        "team_b_rank_date": team_b_rank_date,
        "team_a_model_rank": model_rank_a,
        "team_b_model_rank": model_rank_b,
        "model_rank_date": model_rank_date,
        "team_a_wc_titles": int(feat_dict.get("team_a_wc_titles", 0)),
        "team_b_wc_titles": int(feat_dict.get("team_b_wc_titles", 0)),
        "team_a_avg_goals_wc": round(float(feat_dict.get("team_a_avg_goals_wc", 0)), 2),
        "team_b_avg_goals_wc": round(float(feat_dict.get("team_b_avg_goals_wc", 0)), 2),
        "team_a_clean_sheet_rate": round(float(feat_dict.get("team_a_clean_sheet_rate", 0)), 3),
        "team_b_clean_sheet_rate": round(float(feat_dict.get("team_b_clean_sheet_rate", 0)), 3),
        "team_a_penalty_shootout_wins": int(feat_dict.get("team_a_penalty_shootout_wins", 0)),
        "team_a_penalty_shootout_losses": int(feat_dict.get("team_a_penalty_shootout_losses", 0)),
        "team_b_penalty_shootout_wins": int(feat_dict.get("team_b_penalty_shootout_wins", 0)),
        "team_b_penalty_shootout_losses": int(feat_dict.get("team_b_penalty_shootout_losses", 0)),
        "team_a_top_scorer_wc_goals": int(feat_dict.get("team_a_top_scorer_wc_goals", 0)),
        "team_b_top_scorer_wc_goals": int(feat_dict.get("team_b_top_scorer_wc_goals", 0)),
        "team_a_golden_ball_count": int(feat_dict.get("team_a_golden_ball_count", 0)),
        "team_b_golden_ball_count": int(feat_dict.get("team_b_golden_ball_count", 0)),
        **goal_prediction,
    }
