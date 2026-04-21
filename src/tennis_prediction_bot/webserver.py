from __future__ import annotations

import logging
import os
from threading import Thread

from flask import Flask


LOGGER = logging.getLogger(__name__)
app = Flask(__name__)


@app.get("/")
def home() -> str:
    return "Discord bot is fine"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def should_start_http_server() -> bool:
    if os.getenv("ENABLE_HTTP_SERVER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return bool(os.getenv("PORT"))


def _run_server() -> None:
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, use_reloader=False)


def keep_alive() -> bool:
    if not should_start_http_server():
        LOGGER.info("HTTP keep-alive server disabled. Set PORT or ENABLE_HTTP_SERVER=true to enable it.")
        return False

    thread = Thread(target=_run_server, daemon=True)
    thread.start()
    LOGGER.info("Started HTTP keep-alive server thread.")
    return True
