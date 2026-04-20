from __future__ import annotations

import pandas as pd

from tennis_prediction_bot.features.builder import build_features


def test_build_features_uses_only_prior_matches(tmp_path) -> None:
    cleaned_matches = pd.DataFrame(
        [
            {
                "tourney_id": "2024-0001",
                "tourney_name": "Event 1",
                "surface": "Hard",
                "draw_size": 32,
                "tourney_level": "A",
                "tourney_date": pd.Timestamp("2024-01-01"),
                "match_num": 1,
                "winner_id": 1,
                "winner_name": "Player One",
                "winner_hand": "R",
                "winner_ht": 185,
                "winner_age": 24.0,
                "loser_id": 2,
                "loser_name": "Player Two",
                "loser_hand": "L",
                "loser_ht": 188,
                "loser_age": 25.0,
                "score": "6-4 6-4",
                "best_of": 3,
                "round": "R32",
                "minutes": 90,
                "winner_rank": 10,
                "winner_rank_points": 3000,
                "loser_rank": 20,
                "loser_rank_points": 2000,
                "season": 2024,
                "winner_full_name": "Player One",
                "loser_full_name": "Player Two",
            },
            {
                "tourney_id": "2024-0002",
                "tourney_name": "Event 2",
                "surface": "Hard",
                "draw_size": 32,
                "tourney_level": "A",
                "tourney_date": pd.Timestamp("2024-01-08"),
                "match_num": 1,
                "winner_id": 1,
                "winner_name": "Player One",
                "winner_hand": "R",
                "winner_ht": 185,
                "winner_age": 24.1,
                "loser_id": 2,
                "loser_name": "Player Two",
                "loser_hand": "L",
                "loser_ht": 188,
                "loser_age": 25.1,
                "score": "6-3 6-3",
                "best_of": 3,
                "round": "R32",
                "minutes": 80,
                "winner_rank": 9,
                "winner_rank_points": 3200,
                "loser_rank": 21,
                "loser_rank_points": 1900,
                "season": 2024,
                "winner_full_name": "Player One",
                "loser_full_name": "Player Two",
            },
        ]
    )
    cleaned_data = {"matches": cleaned_matches}

    outputs = build_features(cleaned_data, tmp_path)
    features = outputs["features"]

    second_match_player_one = features.loc[
        (features["match_date"] == pd.Timestamp("2024-01-08"))
        & (features["player_id"] == 1)
        & (features["label"] == 1)
    ].iloc[0]

    assert second_match_player_one["player_matches_before"] == 1
    assert second_match_player_one["player_recent_win_rate"] == 1.0
    assert second_match_player_one["h2h_wins"] == 1
    assert second_match_player_one["player_elo"] == 1516.0
    assert second_match_player_one["opponent_elo"] == 1484.0
    assert second_match_player_one["player_surface_elo"] == 1516.0
    assert second_match_player_one["opponent_surface_elo"] == 1484.0


def test_build_features_creates_player_snapshots(tmp_path) -> None:
    cleaned_matches = pd.DataFrame(
        [
            {
                "tourney_id": "2024-0001",
                "tourney_name": "Event 1",
                "surface": "Clay",
                "draw_size": 32,
                "tourney_level": "A",
                "tourney_date": pd.Timestamp("2024-03-01"),
                "match_num": 1,
                "winner_id": 10,
                "winner_name": "Clay Star",
                "winner_hand": "R",
                "winner_ht": 190,
                "winner_age": 26.2,
                "loser_id": 11,
                "loser_name": "Opponent",
                "loser_hand": "R",
                "loser_ht": 183,
                "loser_age": 27.0,
                "score": "6-1 6-2",
                "best_of": 3,
                "round": "R32",
                "minutes": 61,
                "winner_rank": 15,
                "winner_rank_points": 2200,
                "loser_rank": 40,
                "loser_rank_points": 1100,
                "season": 2024,
                "winner_full_name": "Clay Star",
                "loser_full_name": "Opponent",
            }
        ]
    )

    outputs = build_features({"matches": cleaned_matches}, tmp_path)
    snapshots = outputs["player_snapshots"]
    clay_star = snapshots.loc[snapshots["player_name"] == "Clay Star"].iloc[0]

    assert clay_star["last_rank"] == 15
    assert clay_star["clay_surface_win_rate"] == 1.0
    assert clay_star["overall_elo"] == 1516.0
    assert clay_star["clay_surface_elo"] == 1516.0
