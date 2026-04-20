from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class Settings:
    discord_bot_token: str
    discord_application_id: int | None
    discord_guild_id: int | None
    raw_data_dir: Path
    processed_data_dir: Path
    artifacts_dir: Path
    model_artifact_path: Path
    model_metadata_path: Path
    log_level: str = "INFO"


def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    raw_data_dir = Path(os.getenv("RAW_DATA_DIR", str(PROJECT_ROOT / "raw data")))
    processed_data_dir = Path(
        os.getenv("PROCESSED_DATA_DIR", str(PROJECT_ROOT / "processed_data"))
    )
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))
    model_artifact_path = Path(
        os.getenv("MODEL_ARTIFACT_PATH", str(artifacts_dir / "best_model.joblib"))
    )
    model_metadata_path = Path(
        os.getenv("MODEL_METADATA_PATH", str(artifacts_dir / "model_metadata.json"))
    )

    discord_application_id = os.getenv("DISCORD_APPLICATION_ID", "").strip() or None
    discord_guild_id = os.getenv("DISCORD_GUILD_ID", "").strip() or None

    return Settings(
        discord_bot_token=os.getenv("DISCORD_BOT_TOKEN") or os.getenv("DISCORD_TOKEN", ""),
        discord_application_id=int(discord_application_id) if discord_application_id else None,
        discord_guild_id=int(discord_guild_id) if discord_guild_id else None,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        artifacts_dir=artifacts_dir,
        model_artifact_path=model_artifact_path,
        model_metadata_path=model_metadata_path,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
