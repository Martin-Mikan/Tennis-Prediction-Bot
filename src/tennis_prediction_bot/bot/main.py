from __future__ import annotations

import logging
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from tennis_prediction_bot.config.settings import configure_logging, get_settings
from tennis_prediction_bot.bot.invite import build_invite_url
from tennis_prediction_bot.services.predictor import PredictionService
from tennis_prediction_bot.utils.io import read_json


LOGGER = logging.getLogger(__name__)


class TennisPredictionBot(commands.Bot):
    def __init__(
        self,
        *,
        settings,
        predictor: PredictionService,
        model_metadata: dict[str, object],
    ) -> None:
        intents = discord.Intents.default()
        super().__init__(
            command_prefix="!",
            intents=intents,
            application_id=settings.discord_application_id,
        )
        self.settings = settings
        self.predictor = predictor
        self.model_metadata = model_metadata

    async def setup_hook(self) -> None:
        if self.settings.discord_guild_id:
            guild = discord.Object(id=self.settings.discord_guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            LOGGER.info(
                "Synced %s commands to development guild %s.",
                len(synced),
                self.settings.discord_guild_id,
            )
            return

        synced = await self.tree.sync()
        LOGGER.info("Synced %s global commands.", len(synced))


def _validate_runtime_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise RuntimeError(
            "Missing required runtime files. Train the model first so these files exist:\n"
            f"{joined}"
        )


def build_bot() -> commands.Bot:
    settings = get_settings()
    configure_logging(settings.log_level)
    _validate_runtime_files(
        settings.model_artifact_path,
        settings.model_metadata_path,
        settings.processed_data_dir / "player_snapshots.csv",
        settings.processed_data_dir / "cleaned_matches.csv",
    )

    predictor = PredictionService(
        model_path=settings.model_artifact_path,
        metadata_path=settings.model_metadata_path,
        snapshot_path=settings.processed_data_dir / "player_snapshots.csv",
        matches_path=settings.processed_data_dir / "cleaned_matches.csv",
    )
    model_metadata = read_json(settings.model_metadata_path)
    bot = TennisPredictionBot(
        settings=settings,
        predictor=predictor,
        model_metadata=model_metadata,
    )

    @bot.event
    async def on_ready() -> None:
        LOGGER.info("Bot ready as %s", bot.user)
        if settings.discord_application_id:
            LOGGER.info("Invite URL: %s", build_invite_url(settings.discord_application_id))

    @bot.tree.error
    async def on_app_command_error(
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        LOGGER.exception("Application command failed: %s", error)
        message = "Something went wrong while handling that command."
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)

    @bot.tree.command(name="health", description="Check whether the bot is ready.")
    async def health(interaction: discord.Interaction) -> None:
        await interaction.response.send_message("Bot is online and model artifacts are loaded.")

    @bot.tree.command(name="model_info", description="Show the current model and evaluation metrics.")
    async def model_info(interaction: discord.Interaction) -> None:
        best_model = bot.model_metadata["best_model_name"]
        validation_auc = bot.model_metadata["metrics"][best_model]["validation"]["roc_auc"]
        test_auc = bot.model_metadata["metrics"][best_model]["test"]["roc_auc"]
        message = (
            f"Best model: {best_model}\n"
            f"Validation ROC-AUC: {validation_auc:.3f}\n"
            f"Test ROC-AUC: {test_auc:.3f}"
        )
        await interaction.response.send_message(message)

    @bot.tree.command(name="player", description="Show the latest player snapshot used by the predictor.")
    @app_commands.describe(player_name="ATP player name")
    async def player(interaction: discord.Interaction, player_name: str) -> None:
        try:
            snapshot = bot.predictor.resolve_player(player_name)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return

        message = (
            f"{snapshot['player_name']}\n"
            f"Rank: {int(float(snapshot['last_rank']))}\n"
            f"Rank points: {int(float(snapshot['last_rank_points']))}\n"
            f"Recent win rate: {float(snapshot['recent_win_rate']):.1%}\n"
            f"Career win rate: {float(snapshot['career_win_rate']):.1%}"
        )
        await interaction.response.send_message(message)

    @bot.tree.command(name="predict", description="Predict the winner of a hypothetical ATP match.")
    @app_commands.describe(
        player_a="First player name",
        player_b="Second player name",
        surface="Optional surface: Hard, Clay, Grass, or Carpet",
    )
    async def predict(
        interaction: discord.Interaction,
        player_a: str,
        player_b: str,
        surface: str = "Hard",
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            result = bot.predictor.predict_match(player_a, player_b, surface)
        except ValueError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return

        message = (
            f"Predicted winner: {result.winner_name}\n"
            f"Confidence: {result.confidence:.1%}\n"
            f"{player_a}: {result.player_a_win_probability:.1%}\n"
            f"{player_b}: {result.player_b_win_probability:.1%}\n"
            f"{player_a} {surface.title()} win rate: {result.player_a_surface_win_rate:.1%}\n"
            f"{player_b} {surface.title()} win rate: {result.player_b_surface_win_rate:.1%}\n"
            f"Head-to-head: {player_a} {result.h2h_player_a_wins} - {result.h2h_player_b_wins} {player_b}\n"
            f"{result.summary}"
        )
        await interaction.followup.send(message)

    return bot


def main() -> None:
    settings = get_settings()
    if not settings.discord_bot_token:
        raise RuntimeError("DISCORD_BOT_TOKEN is missing. Set it in your environment or .env file.")
    bot = build_bot()
    bot.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
