from __future__ import annotations

from pathlib import Path

import pandas as pd

from tennis_prediction_bot.utils.io import ensure_directory


MATCH_YEARS = list(range(2000, 2025))
MATCH_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_age",
    "loser_id",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_age",
    "score",
    "best_of",
    "round",
    "minutes",
    "winner_rank",
    "winner_rank_points",
    "loser_rank",
    "loser_rank_points",
]


def _match_paths(raw_data_dir: Path) -> list[Path]:
    return [raw_data_dir / f"atp_matches_{year}.csv" for year in MATCH_YEARS]


def load_matches(raw_data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _match_paths(raw_data_dir):
        frame = pd.read_csv(path, low_memory=False)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_players(raw_data_dir: Path) -> pd.DataFrame:
    players = pd.read_csv(raw_data_dir / "atp_players.csv", low_memory=False)
    players = players.rename(
        columns={
            "name_first": "first_name",
            "name_last": "last_name",
            "dob": "birth_date",
            "ioc": "country_code",
            "height": "height_cm",
        }
    )
    players["full_name"] = (
        players["first_name"].fillna("").str.strip()
        + " "
        + players["last_name"].fillna("").str.strip()
    ).str.strip()
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce").astype("Int64")
    return players


def load_rankings(raw_data_dir: Path) -> pd.DataFrame:
    frames = []
    for name in [
        "atp_rankings_00s.csv",
        "atp_rankings_10s.csv",
        "atp_rankings_20s.csv",
        "atp_rankings_current.csv",
    ]:
        frame = pd.read_csv(raw_data_dir / name, low_memory=False)
        frame.columns = ["ranking_date", "rank", "player_id", "points"]
        frames.append(frame)
    rankings = pd.concat(frames, ignore_index=True).drop_duplicates()
    rankings["ranking_date"] = pd.to_datetime(rankings["ranking_date"], format="%Y%m%d", errors="coerce")
    rankings["player_id"] = pd.to_numeric(rankings["player_id"], errors="coerce").astype("Int64")
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="coerce")
    rankings["points"] = pd.to_numeric(rankings["points"], errors="coerce")
    return rankings.sort_values(["ranking_date", "rank", "player_id"]).reset_index(drop=True)


def _normalize_matches(matches: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d", errors="coerce")

    numeric_columns = [
        "winner_id",
        "loser_id",
        "winner_ht",
        "loser_ht",
        "winner_age",
        "loser_age",
        "winner_rank",
        "loser_rank",
        "winner_rank_points",
        "loser_rank_points",
        "minutes",
        "match_num",
        "draw_size",
        "best_of",
    ]
    for column in numeric_columns:
        matches[column] = pd.to_numeric(matches[column], errors="coerce")

    matches["winner_id"] = matches["winner_id"].astype("Int64")
    matches["loser_id"] = matches["loser_id"].astype("Int64")
    matches["surface"] = matches["surface"].fillna("Unknown").astype(str)
    matches["tourney_level"] = matches["tourney_level"].fillna("A").astype(str)
    matches["round"] = matches["round"].fillna("Unknown").astype(str)
    matches["winner_name"] = matches["winner_name"].fillna("").astype(str).str.strip()
    matches["loser_name"] = matches["loser_name"].fillna("").astype(str).str.strip()
    matches["score"] = matches["score"].fillna("").astype(str)

    invalid_score_pattern = r"RET|W/O|DEF|ABN|Walkover"
    valid_mask = (
        matches["winner_id"].notna()
        & matches["loser_id"].notna()
        & matches["winner_name"].ne("")
        & matches["loser_name"].ne("")
        & matches["tourney_date"].notna()
        & ~matches["score"].str.contains(invalid_score_pattern, case=False, na=False, regex=True)
    )
    cleaned = matches.loc[valid_mask, MATCH_COLUMNS].copy()
    cleaned["season"] = cleaned["tourney_date"].dt.year.astype(int)

    player_lookup = players[["player_id", "full_name"]].dropna().drop_duplicates()
    cleaned = cleaned.merge(
        player_lookup.rename(columns={"player_id": "winner_id", "full_name": "winner_full_name"}),
        on="winner_id",
        how="left",
    )
    cleaned = cleaned.merge(
        player_lookup.rename(columns={"player_id": "loser_id", "full_name": "loser_full_name"}),
        on="loser_id",
        how="left",
    )
    cleaned["winner_full_name"] = cleaned["winner_full_name"].fillna(cleaned["winner_name"])
    cleaned["loser_full_name"] = cleaned["loser_full_name"].fillna(cleaned["loser_name"])

    return cleaned.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)


def clean_raw_data(raw_data_dir: Path, processed_data_dir: Path) -> dict[str, pd.DataFrame]:
    ensure_directory(processed_data_dir)

    players = load_players(raw_data_dir)
    rankings = load_rankings(raw_data_dir)
    matches = load_matches(raw_data_dir)
    cleaned_matches = _normalize_matches(matches, players)

    cleaned_matches.to_csv(processed_data_dir / "cleaned_matches.csv", index=False)
    rankings.to_csv(processed_data_dir / "cleaned_rankings.csv", index=False)
    players.to_csv(processed_data_dir / "cleaned_players.csv", index=False)

    return {
        "matches": cleaned_matches,
        "players": players,
        "rankings": rankings,
    }
