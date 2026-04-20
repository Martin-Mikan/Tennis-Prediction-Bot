from __future__ import annotations

import json

import joblib
import pandas as pd

from tennis_prediction_bot.services.predictor import PredictionService


class StubModel:
    def predict_proba(self, frame):
        player_rank = float(frame.iloc[0]["player_rank"])
        opponent_rank = float(frame.iloc[0]["opponent_rank"])
        probability = 0.8 if player_rank < opponent_rank else 0.2
        return [[1.0 - probability, probability]]


def test_prediction_service_prefers_better_ranked_player(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    snapshot_path = tmp_path / "snapshots.csv"
    matches_path = tmp_path / "matches.csv"

    joblib.dump(StubModel(), model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "numeric_columns": [
                    "player_rank",
                    "opponent_rank",
                    "rank_diff",
                    "player_rank_points",
                    "opponent_rank_points",
                    "rank_points_diff",
                    "player_age",
                    "opponent_age",
                    "age_diff",
                    "player_height",
                    "opponent_height",
                    "height_diff",
                    "player_elo",
                    "opponent_elo",
                    "elo_diff",
                    "player_surface_elo",
                    "opponent_surface_elo",
                    "surface_elo_diff",
                    "player_matches_before",
                    "opponent_matches_before",
                    "player_win_rate",
                    "opponent_win_rate",
                    "recent_form_diff",
                    "player_recent_win_rate",
                    "opponent_recent_win_rate",
                    "player_surface_win_rate",
                    "opponent_surface_win_rate",
                    "surface_win_rate_diff",
                    "h2h_wins",
                    "h2h_losses",
                    "h2h_diff",
                ],
                "categorical_columns": [
                    "surface",
                    "tourney_level",
                    "player_hand",
                    "opponent_hand",
                    "handedness_matchup",
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Top Player",
                "last_rank": 5,
                "last_rank_points": 5000,
                "last_age": 24.0,
                "last_height": 188,
                "overall_elo": 1620.0,
                "hand": "R",
                "career_matches": 100,
                "career_win_rate": 0.7,
                "recent_win_rate": 0.8,
                "hard_surface_win_rate": 0.75,
                "hard_surface_elo": 1650.0,
                "clay_surface_win_rate": 0.6,
                "clay_surface_elo": 1540.0,
                "grass_surface_win_rate": 0.7,
                "grass_surface_elo": 1585.0,
                "carpet_surface_win_rate": 0.0,
                "carpet_surface_elo": 1500.0,
            },
            {
                "player_id": 2,
                "player_name": "Lower Player",
                "last_rank": 50,
                "last_rank_points": 1200,
                "last_age": 28.0,
                "last_height": 182,
                "overall_elo": 1450.0,
                "hand": "L",
                "career_matches": 90,
                "career_win_rate": 0.45,
                "recent_win_rate": 0.4,
                "hard_surface_win_rate": 0.4,
                "hard_surface_elo": 1430.0,
                "clay_surface_win_rate": 0.42,
                "clay_surface_elo": 1470.0,
                "grass_surface_win_rate": 0.3,
                "grass_surface_elo": 1410.0,
                "carpet_surface_win_rate": 0.0,
                "carpet_surface_elo": 1500.0,
            },
        ]
    ).to_csv(snapshot_path, index=False)
    pd.DataFrame(
        [
            {"winner_id": 1, "loser_id": 2},
            {"winner_id": 1, "loser_id": 2},
            {"winner_id": 2, "loser_id": 1},
        ]
    ).to_csv(matches_path, index=False)

    service = PredictionService(model_path, metadata_path, snapshot_path, matches_path)
    result = service.predict_match("Top Player", "Lower Player", "Hard")

    assert result.winner_name == "Top Player"
    assert result.confidence > 0.5
    assert result.player_a_surface_win_rate == 0.75
    assert result.player_b_surface_win_rate == 0.4
    assert result.h2h_player_a_wins == 2
    assert result.h2h_player_b_wins == 1
