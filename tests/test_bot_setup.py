from __future__ import annotations

from tennis_prediction_bot.bot.invite import build_invite_url


def test_build_invite_url_contains_expected_scopes() -> None:
    url = build_invite_url(123456789012345678)

    assert "client_id=123456789012345678" in url
    assert "scope=bot+" in url or "scope=bot%20" in url
    assert "applications.commands" in url
