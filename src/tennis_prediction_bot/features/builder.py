from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tennis_prediction_bot.utils.io import ensure_directory


SURFACE_KEYS = ("Hard", "Clay", "Grass", "Carpet", "Unknown")
RECENT_WINDOW = 10
BASE_ELO = 1500.0
ELO_K_FACTOR = 32.0


@dataclass
class PlayerState:
    total_matches: int = 0
    total_wins: int = 0
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_rank: float | None = None
    last_rank_points: float | None = None
    last_age: float | None = None
    last_height: float | None = None
    elo: float = BASE_ELO
    surface_elo: dict[str, float] = field(
        default_factory=lambda: defaultdict(lambda: BASE_ELO)
    )
    hand: str = "U"
    name: str = ""


def _safe_ratio(num: float, den: float) -> float:
    if not den:
        return 0.0
    return float(num) / float(den)


def _expected_score(player_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))


def _build_row(
    focal_id: int,
    opponent_id: int,
    surface: str,
    tourney_level: str,
    match_date: pd.Timestamp,
    focal_name: str,
    opponent_name: str,
    focal_state: PlayerState,
    opponent_state: PlayerState,
    focal_rank: float,
    opponent_rank: float,
    focal_points: float,
    opponent_points: float,
    focal_age: float,
    opponent_age: float,
    focal_height: float,
    opponent_height: float,
    focal_elo: float,
    opponent_elo: float,
    focal_surface_elo: float,
    opponent_surface_elo: float,
    h2h_wins: int,
    h2h_losses: int,
    label: int,
) -> dict[str, object]:
    recent_focal = sum(focal_state.recent_results)
    recent_opponent = sum(opponent_state.recent_results)
    focal_surface_matches = focal_state.surface_matches[surface]
    opponent_surface_matches = opponent_state.surface_matches[surface]

    return {
        "match_date": match_date,
        "season": match_date.year,
        "surface": surface,
        "tourney_level": tourney_level,
        "player_id": focal_id,
        "player_name": focal_name,
        "opponent_id": opponent_id,
        "opponent_name": opponent_name,
        "player_rank": focal_rank,
        "opponent_rank": opponent_rank,
        "rank_diff": opponent_rank - focal_rank,
        "player_rank_points": focal_points,
        "opponent_rank_points": opponent_points,
        "rank_points_diff": focal_points - opponent_points,
        "player_age": focal_age,
        "opponent_age": opponent_age,
        "age_diff": focal_age - opponent_age,
        "player_height": focal_height,
        "opponent_height": opponent_height,
        "height_diff": focal_height - opponent_height,
        "player_elo": focal_elo,
        "opponent_elo": opponent_elo,
        "elo_diff": focal_elo - opponent_elo,
        "player_surface_elo": focal_surface_elo,
        "opponent_surface_elo": opponent_surface_elo,
        "surface_elo_diff": focal_surface_elo - opponent_surface_elo,
        "player_hand": focal_state.hand or "U",
        "opponent_hand": opponent_state.hand or "U",
        "handedness_matchup": f"{focal_state.hand or 'U'}_vs_{opponent_state.hand or 'U'}",
        "player_matches_before": focal_state.total_matches,
        "opponent_matches_before": opponent_state.total_matches,
        "player_win_rate": _safe_ratio(focal_state.total_wins, focal_state.total_matches),
        "opponent_win_rate": _safe_ratio(opponent_state.total_wins, opponent_state.total_matches),
        "recent_form_diff": _safe_ratio(recent_focal, len(focal_state.recent_results))
        - _safe_ratio(recent_opponent, len(opponent_state.recent_results)),
        "player_recent_win_rate": _safe_ratio(recent_focal, len(focal_state.recent_results)),
        "opponent_recent_win_rate": _safe_ratio(recent_opponent, len(opponent_state.recent_results)),
        "player_surface_win_rate": _safe_ratio(
            focal_state.surface_wins[surface], focal_surface_matches
        ),
        "opponent_surface_win_rate": _safe_ratio(
            opponent_state.surface_wins[surface], opponent_surface_matches
        ),
        "surface_win_rate_diff": _safe_ratio(
            focal_state.surface_wins[surface], focal_surface_matches
        )
        - _safe_ratio(opponent_state.surface_wins[surface], opponent_surface_matches),
        "h2h_wins": h2h_wins,
        "h2h_losses": h2h_losses,
        "h2h_diff": h2h_wins - h2h_losses,
        "label": label,
    }


def build_features(cleaned_data: dict[str, pd.DataFrame], processed_data_dir: Path) -> dict[str, pd.DataFrame]:
    ensure_directory(processed_data_dir)

    matches = cleaned_data["matches"].sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)

    states: dict[int, PlayerState] = defaultdict(PlayerState)
    head_to_head: dict[tuple[int, int], int] = defaultdict(int)
    feature_rows: list[dict[str, object]] = []

    for match in matches.itertuples(index=False):
        surface = match.surface if match.surface in SURFACE_KEYS else "Unknown"
        winner_id = int(match.winner_id)
        loser_id = int(match.loser_id)

        winner_state = states[winner_id]
        loser_state = states[loser_id]
        winner_state.hand = str(match.winner_hand or "U")
        loser_state.hand = str(match.loser_hand or "U")
        winner_state.name = str(match.winner_full_name)
        loser_state.name = str(match.loser_full_name)

        winner_rank = float(match.winner_rank) if pd.notna(match.winner_rank) else 9999.0
        loser_rank = float(match.loser_rank) if pd.notna(match.loser_rank) else 9999.0
        winner_points = float(match.winner_rank_points) if pd.notna(match.winner_rank_points) else 0.0
        loser_points = float(match.loser_rank_points) if pd.notna(match.loser_rank_points) else 0.0
        winner_age = float(match.winner_age) if pd.notna(match.winner_age) else 0.0
        loser_age = float(match.loser_age) if pd.notna(match.loser_age) else 0.0
        winner_height = float(match.winner_ht) if pd.notna(match.winner_ht) else 0.0
        loser_height = float(match.loser_ht) if pd.notna(match.loser_ht) else 0.0

        winner_pair = (winner_id, loser_id)
        loser_pair = (loser_id, winner_id)
        winner_elo = winner_state.elo
        loser_elo = loser_state.elo
        winner_surface_elo = winner_state.surface_elo[surface]
        loser_surface_elo = loser_state.surface_elo[surface]

        feature_rows.append(
            _build_row(
                focal_id=winner_id,
                opponent_id=loser_id,
                surface=surface,
                tourney_level=match.tourney_level,
                match_date=match.tourney_date,
                focal_name=match.winner_full_name,
                opponent_name=match.loser_full_name,
                focal_state=winner_state,
                opponent_state=loser_state,
                focal_rank=winner_rank,
                opponent_rank=loser_rank,
                focal_points=winner_points,
                opponent_points=loser_points,
                focal_age=winner_age,
                opponent_age=loser_age,
                focal_height=winner_height,
                opponent_height=loser_height,
                focal_elo=winner_elo,
                opponent_elo=loser_elo,
                focal_surface_elo=winner_surface_elo,
                opponent_surface_elo=loser_surface_elo,
                h2h_wins=head_to_head[winner_pair],
                h2h_losses=head_to_head[loser_pair],
                label=1,
            )
        )
        feature_rows.append(
            _build_row(
                focal_id=loser_id,
                opponent_id=winner_id,
                surface=surface,
                tourney_level=match.tourney_level,
                match_date=match.tourney_date,
                focal_name=match.loser_full_name,
                opponent_name=match.winner_full_name,
                focal_state=loser_state,
                opponent_state=winner_state,
                focal_rank=loser_rank,
                opponent_rank=winner_rank,
                focal_points=loser_points,
                opponent_points=winner_points,
                focal_age=loser_age,
                opponent_age=winner_age,
                focal_height=loser_height,
                opponent_height=winner_height,
                focal_elo=loser_elo,
                opponent_elo=winner_elo,
                focal_surface_elo=loser_surface_elo,
                opponent_surface_elo=winner_surface_elo,
                h2h_wins=head_to_head[loser_pair],
                h2h_losses=head_to_head[winner_pair],
                label=0,
            )
        )

        winner_state.total_matches += 1
        loser_state.total_matches += 1
        winner_state.total_wins += 1
        winner_state.recent_results.append(1)
        loser_state.recent_results.append(0)
        winner_state.surface_matches[surface] += 1
        loser_state.surface_matches[surface] += 1
        winner_state.surface_wins[surface] += 1
        head_to_head[winner_pair] += 1

        expected_winner = _expected_score(winner_elo, loser_elo)
        expected_winner_surface = _expected_score(winner_surface_elo, loser_surface_elo)
        winner_state.elo = winner_elo + ELO_K_FACTOR * (1.0 - expected_winner)
        loser_state.elo = loser_elo + ELO_K_FACTOR * (0.0 - (1.0 - expected_winner))
        winner_state.surface_elo[surface] = (
            winner_surface_elo + ELO_K_FACTOR * (1.0 - expected_winner_surface)
        )
        loser_state.surface_elo[surface] = (
            loser_surface_elo + ELO_K_FACTOR * (0.0 - (1.0 - expected_winner_surface))
        )

        winner_state.last_rank = winner_rank
        loser_state.last_rank = loser_rank
        winner_state.last_rank_points = winner_points
        loser_state.last_rank_points = loser_points
        winner_state.last_age = winner_age
        loser_state.last_age = loser_age
        winner_state.last_height = winner_height
        loser_state.last_height = loser_height

    features = pd.DataFrame(feature_rows).sort_values(["match_date", "player_id", "opponent_id"]).reset_index(drop=True)
    features.to_csv(processed_data_dir / "training_features.csv", index=False)

    player_snapshot_rows = []
    for player_id, state in states.items():
        row = {
            "player_id": player_id,
            "player_name": state.name,
            "last_rank": state.last_rank if state.last_rank is not None else 9999.0,
            "last_rank_points": state.last_rank_points if state.last_rank_points is not None else 0.0,
            "last_age": state.last_age if state.last_age is not None else 0.0,
            "last_height": state.last_height if state.last_height is not None else 0.0,
            "overall_elo": state.elo,
            "hand": state.hand or "U",
            "career_matches": state.total_matches,
            "career_win_rate": _safe_ratio(state.total_wins, state.total_matches),
            "recent_win_rate": _safe_ratio(sum(state.recent_results), len(state.recent_results)),
        }
        for surface in SURFACE_KEYS:
            row[f"{surface.lower()}_surface_win_rate"] = _safe_ratio(
                state.surface_wins[surface], state.surface_matches[surface]
            )
            row[f"{surface.lower()}_surface_elo"] = state.surface_elo[surface]
        player_snapshot_rows.append(row)

    player_snapshots = pd.DataFrame(player_snapshot_rows).sort_values("player_name").reset_index(drop=True)
    player_snapshots.to_csv(processed_data_dir / "player_snapshots.csv", index=False)

    return {
        "features": features,
        "player_snapshots": player_snapshots,
    }
