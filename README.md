# Tennis Prediction Bot

A brand-new Python project for training an ATP main-tour tennis match predictor and serving predictions through a Discord slash-command bot.

## What It Does

- Cleans Jeff Sackmann ATP raw CSV files from `raw data`
- Builds pre-match features without future leakage
- Trains two models:
  - logistic regression baseline
  - boosted tree models including LightGBM
- Saves the best model, evaluation metrics, and feature metadata
- Exposes Discord slash commands:
  - `/predict`
  - `/player`
  - `/model_info`
  - `/health`

## Setup

1. Install Python 3.11+.
2. Create and activate a virtual environment.
3. Install the package:

```bash
pip install -e .[dev]
```

4. Copy `.env.example` to `.env` and set your Discord token.

## Discord Bot Setup

Follow the Discord application setup flow described in the discord.py docs:

1. Open the Discord Developer Portal and create a new application.
2. In the `Bot` section, create the bot user and copy the bot token into `.env` as `DISCORD_BOT_TOKEN`.
3. Copy the application ID into `.env` as `DISCORD_APPLICATION_ID`.
4. For faster slash-command testing, add one server ID as `DISCORD_GUILD_ID`.
5. Keep the default intents for this bot. This project uses slash commands and does not require privileged message-content intent.
6. Generate the invite URL:

```bash
tennis-bot-invite
```

7. Open the printed URL and add the bot to your Discord server.

If `DISCORD_GUILD_ID` is set, commands are synced to that development server on startup for faster iteration. In production, automatic global sync is disabled by default to avoid unnecessary Discord API traffic. If you ever need a one-time forced global sync, set `DISCORD_SYNC_COMMANDS_ON_STARTUP=true` for that deploy only.

## Train The Model

```bash
tennis-bot-train
```

## Run The Discord Bot

```bash
tennis-bot-run
```

Before running the bot, make sure training has already generated:

- `artifacts/best_model.joblib`
- `artifacts/model_metadata.json`
- `processed_data/player_snapshots.csv`

The bot now validates those files at startup and will stop with a clear error if they are missing.

## Render Web Service With Uptime Robot

If you want to run this as a Render Web Service instead of a background worker, the project now includes a tiny Flask server for `/` and `/health`.

How it works:

- If Render provides `PORT`, the Flask keep-alive server starts automatically
- The Discord bot still runs in the same process
- You can point Uptime Robot at the Render URL to keep the free web service warm

Recommended Render Web Service settings:

- Build Command: `pip install .`
- Start Command: `python -m tennis_prediction_bot.bot.main`

Recommended environment variables:

- `DISCORD_TOKEN`
- `LOG_LEVEL=INFO`
- `DISCORD_SYNC_COMMANDS_ON_STARTUP=false`

Optional local override:

- `ENABLE_HTTP_SERVER=true`

Uptime Robot can ping either `/` or `/health`.

## GitHub To Render

Yes, the intended deployment flow is:

1. Push this project to GitHub.
2. Import that GitHub repository into Render.
3. Let Render create the worker using `render.yaml`.
4. Add `DISCORD_TOKEN` in Render as a secret environment variable.
5. Deploy.

This repo now includes:

- `.gitignore` so secrets, logs, local virtualenvs, and large local-only files stay out of GitHub
- `.github/workflows/ci.yml` so GitHub runs tests on pushes and pull requests
- `render.yaml` so Render can run the bot as a 24/7 background worker

The runtime files that should stay in the repo for Render are:

- `artifacts/best_model.joblib`
- `artifacts/model_metadata.json`
- `artifacts/evaluation_metrics.json`
- `artifacts/training_summary.json`
- `processed_data/cleaned_matches.csv`
- `processed_data/player_snapshots.csv`

The local `.env`, the virtual environment, logs, and the large raw training data should not go to GitHub.

## Notes

- v1 intentionally excludes challengers, qualifying, futures, and amateur events
- predictions are statistical estimates, not guarantees
- raw data attribution: Jeff Sackmann / Tennis Abstract
