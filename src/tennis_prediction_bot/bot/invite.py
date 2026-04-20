from __future__ import annotations

import discord

from tennis_prediction_bot.config.settings import get_settings


def build_invite_url(application_id: int) -> str:
    permissions = discord.Permissions(
        view_channel=True,
        send_messages=True,
        read_message_history=True,
    )
    return discord.utils.oauth_url(
        application_id,
        permissions=permissions,
        scopes=("bot", "applications.commands"),
    )


def main() -> None:
    settings = get_settings()
    if not settings.discord_application_id:
        raise RuntimeError(
            "DISCORD_APPLICATION_ID is missing. Set it in .env before generating an invite URL."
        )
    print(build_invite_url(settings.discord_application_id))


if __name__ == "__main__":
    main()
