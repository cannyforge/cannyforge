"""
CannyForge Slack Service

Sends messages to Slack channels. Operates in mock mode when SLACK_BOT_TOKEN
is not set, real mode when it is.

Usage:
    service = SlackService()
    service.connect()
    response = service.send_message(channel="#general", text="Hello from CannyForge!")
"""

import logging
import os
from cannyforge.services.service_base import ServiceResponse

logger = logging.getLogger("SlackService")


class SlackService:
    """Slack integration with mock/real mode based on API key presence."""

    def __init__(self):
        self._token = os.environ.get("SLACK_BOT_TOKEN")
        self._mock = self._token is None
        self._client = None

    @property
    def is_mock(self) -> bool:
        return self._mock

    def connect(self):
        if self._mock:
            logger.info("Slack: running in mock mode (set SLACK_BOT_TOKEN for real)")
            return
        try:
            from slack_sdk import WebClient
            self._client = WebClient(token=self._token)
            logger.info("Slack: connected with real token")
        except ImportError:
            logger.warning("slack_sdk not installed, falling back to mock mode")
            self._mock = True

    def send_message(self, channel: str, text: str) -> ServiceResponse:
        if self._mock:
            return ServiceResponse(
                success=True,
                data={"channel": channel, "text": text, "ts": "mock_1234567890.000100", "mock": True},
            )
        try:
            result = self._client.chat_postMessage(channel=channel, text=text)
            return ServiceResponse(
                success=True,
                data={"channel": result["channel"], "ts": result["ts"]},
            )
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    def list_channels(self) -> ServiceResponse:
        if self._mock:
            return ServiceResponse(
                success=True,
                data={"channels": ["#general", "#random", "#engineering"], "mock": True},
            )
        try:
            result = self._client.conversations_list()
            channels = [c["name"] for c in result["channels"]]
            return ServiceResponse(success=True, data={"channels": channels})
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))
