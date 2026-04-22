from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from tennis_prediction_bot.utils.io import read_json


SURFACE_COLUMN_MAP = {
    "hard": "hard_surface_win_rate",
    "clay": "clay_surface_win_rate",
    "grass": "grass_surface_win_rate",
    "carpet": "carpet_surface_win_rate",
}
SURFACE_ELO_COLUMN_MAP = {
    "hard": "hard_surface_elo",
    "clay": "clay_surface_elo",
    "grass": "grass_surface_elo",
    "carpet": "carpet_surface_elo",
}


@dataclass(slots=True)
class PredictionResult:
    winner_name: str
    confidence: float
    player_a_win_probability: float
    player_b_win_probability: float
    player_a_surface_win_rate: float
    player_b_surface_win_rate: float
    h2h_player_a_wins: int
    h2h_player_b_wins: int
    summary: str


class PredictionService:
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        snapshot_path: Path,
        matches_path: Path,
    ) -> None:
        self.model = joblib.load(model_path)
        self.metadata = read_json(metadata_path)
        self.snapshots = pd.read_csv(snapshot_path)
        self.snapshots["search_name"] = self.snapshots["player_name"].str.lower().str.strip()
        self.head_to_head = self._load_head_to_head(matches_path)

    def _load_head_to_head(self, matches_path: Path) -> dict[tuple[int, int], int]:
        matches = pd.read_csv(
            matches_path,
            usecols=["winner_id", "loser_id"],
            low_memory=False,
        )
        grouped = (
            matches.groupby(["winner_id", "loser_id"])
            .size()
            .reset_index(name="wins")
        )
        return {
            (int(row["winner_id"]), int(row["loser_id"])): int(row["wins"])
            for _, row in grouped.iterrows()
        }

    def get_head_to_head(self, player_id: int, opponent_id: int) -> tuple[int, int]:
        return (
            self.head_to_head.get((player_id, opponent_id), 0),
            self.head_to_head.get((opponent_id, player_id), 0),
        )

    def resolve_player(self, player_name: str) -> dict[str, Any]:
        normalized = player_name.lower().strip()
        exact = self.snapshots.loc[self.snapshots["search_name"] == normalized]
        if len(exact) == 1:
            return exact.iloc[0].to_dict()

        contains = self.snapshots.loc[self.snapshots["search_name"].str.contains(normalized, regex=False)]
        if len(contains) == 1:
            return contains.iloc[0].to_dict()
        if len(contains) == 0:
            raise ValueError(f"Could not find player '{player_name}'.")

        sample = ", ".join(contains["player_name"].head(5).tolist())
        raise ValueError(f"Player name '{player_name}' is ambiguous. Matches: {sample}")

    def _build_feature_row(
        self,
        player: dict[str, Any],
        opponent: dict[str, Any],
        surface: str,
    ) -> pd.DataFrame:
        normalized_surface = surface.title()
        surface_column = SURFACE_COLUMN_MAP.get(surface.lower(), "hard_surface_win_rate")
        surface_elo_column = SURFACE_ELO_COLUMN_MAP.get(surface.lower(), "hard_surface_elo")

        player_surface = float(player.get(surface_column, 0.0))
        opponent_surface = float(opponent.get(surface_column, 0.0))
        player_surface_elo = float(player.get(surface_elo_column, 1500.0))
        opponent_surface_elo = float(opponent.get(surface_elo_column, 1500.0))
        player_recent = float(player.get("recent_win_rate", 0.0))
        opponent_recent = float(opponent.get("recent_win_rate", 0.0))
        player_elo = float(player.get("overall_elo", 1500.0))
        opponent_elo = float(opponent.get("overall_elo", 1500.0))
        h2h_wins, h2h_losses = self.get_head_to_head(
            int(player["player_id"]),
            int(opponent["player_id"]),
        )

        row = {
            "player_rank": float(player.get("last_rank", 9999.0)),
            "opponent_rank": float(opponent.get("last_rank", 9999.0)),
            "rank_diff": float(opponent.get("last_rank", 9999.0)) - float(player.get("last_rank", 9999.0)),
            "player_rank_points": float(player.get("last_rank_points", 0.0)),
            "opponent_rank_points": float(opponent.get("last_rank_points", 0.0)),
            "rank_points_diff": float(player.get("last_rank_points", 0.0)) - float(opponent.get("last_rank_points", 0.0)),
            "player_age": float(player.get("last_age", 0.0)),
            "opponent_age": float(opponent.get("last_age", 0.0)),
            "age_diff": float(player.get("last_age", 0.0)) - float(opponent.get("last_age", 0.0)),
            "player_height": float(player.get("last_height", 0.0)),
            "opponent_height": float(opponent.get("last_height", 0.0)),
            "height_diff": float(player.get("last_height", 0.0)) - float(opponent.get("last_height", 0.0)),
            "player_elo": player_elo,
            "opponent_elo": opponent_elo,
            "elo_diff": player_elo - opponent_elo,
            "player_surface_elo": player_surface_elo,
            "opponent_surface_elo": opponent_surface_elo,
            "surface_elo_diff": player_surface_elo - opponent_surface_elo,
            "player_matches_before": float(player.get("career_matches", 0.0)),
            "opponent_matches_before": float(opponent.get("career_matches", 0.0)),
            "player_win_rate": float(player.get("career_win_rate", 0.0)),
            "opponent_win_rate": float(opponent.get("career_win_rate", 0.0)),
            "recent_form_diff": player_recent - opponent_recent,
            "player_recent_win_rate": player_recent,
            "opponent_recent_win_rate": opponent_recent,
            "player_surface_win_rate": player_surface,
            "opponent_surface_win_rate": opponent_surface,
            "surface_win_rate_diff": player_surface - opponent_surface,
            "h2h_wins": float(h2h_wins),
            "h2h_losses": float(h2h_losses),
            "h2h_diff": float(h2h_wins - h2h_losses),
            "surface": normalized_surface,
            "tourney_level": "A",
            "player_hand": str(player.get("hand", "U")),
            "opponent_hand": str(opponent.get("hand", "U")),
            "handedness_matchup": f"{player.get('hand', 'U')}_vs_{opponent.get('hand', 'U')}",
        }
        ordered = self.metadata["numeric_columns"] + self.metadata["categorical_columns"]
        return pd.DataFrame([[row[column] for column in ordered]], columns=ordered)

    def predict_match(self, player_a: str, player_b: str, surface: str = "Hard") -> PredictionResult:
        player_a_snapshot = self.resolve_player(player_a)
        player_b_snapshot = self.resolve_player(player_b)
        surface_column = SURFACE_COLUMN_MAP.get(surface.lower(), "hard_surface_win_rate")
        player_a_surface_win_rate = float(player_a_snapshot.get(surface_column, 0.0))
        player_b_surface_win_rate = float(player_b_snapshot.get(surface_column, 0.0))
        h2h_player_a_wins, h2h_player_b_wins = self.get_head_to_head(
            int(player_a_snapshot["player_id"]),
            int(player_b_snapshot["player_id"]),
        )

        a_frame = self._build_feature_row(player_a_snapshot, player_b_snapshot, surface)
        b_frame = self._build_feature_row(player_b_snapshot, player_a_snapshot, surface)
        a_probability = float(self.model.predict_proba(a_frame)[0][1])
        b_probability = float(self.model.predict_proba(b_frame)[0][1])

        if a_probability >= b_probability:
            winner_name = str(player_a_snapshot["player_name"])
            confidence = a_probability
        else:
            winner_name = str(player_b_snapshot["player_name"])
            confidence = b_probability

        summary = (
            f"{winner_name} will win this shit on {surface.title()} "
            f"dựa trên nguồn trust me bro. "
            "Dự đoán mang tính hên xui, bet thua cấm đổ thừa đcm"
        )
        return PredictionResult(
            winner_name=winner_name,
            confidence=confidence,
            player_a_win_probability=a_probability,
            player_b_win_probability=b_probability,
            player_a_surface_win_rate=player_a_surface_win_rate,
            player_b_surface_win_rate=player_b_surface_win_rate,
            h2h_player_a_wins=h2h_player_a_wins,
            h2h_player_b_wins=h2h_player_b_wins,
            summary=summary,
        )
