"""
CannyForge CRM Service

Basic CRM integration for contact lookup and activity logging.
Mock mode when no API key is set.

Usage:
    service = CRMService()
    service.connect()
    response = service.lookup_contact(email="user@example.com")
"""

import logging
import os
from cannyforge.services.service_base import ServiceResponse

logger = logging.getLogger("CRMService")


class CRMService:
    """CRM integration with mock/real mode."""

    def __init__(self):
        self._api_key = os.environ.get("CRM_API_KEY")
        self._api_url = os.environ.get("CRM_API_URL", "")
        self._mock = self._api_key is None

    @property
    def is_mock(self) -> bool:
        return self._mock

    def connect(self):
        if self._mock:
            logger.info("CRM: running in mock mode (set CRM_API_KEY for real)")
            return
        logger.info(f"CRM: connected to {self._api_url}")

    def lookup_contact(self, email: str) -> ServiceResponse:
        if self._mock:
            return ServiceResponse(
                success=True,
                data={
                    "email": email,
                    "name": "Mock Contact",
                    "company": "Mock Corp",
                    "last_activity": "2024-01-15",
                    "mock": True,
                },
            )
        try:
            import requests
            resp = requests.get(
                f"{self._api_url}/contacts",
                params={"email": email},
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            return ServiceResponse(success=True, data=resp.json())
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    def log_activity(self, contact_email: str, activity_type: str,
                     description: str) -> ServiceResponse:
        if self._mock:
            return ServiceResponse(
                success=True,
                data={
                    "contact": contact_email,
                    "activity_type": activity_type,
                    "description": description,
                    "id": "mock_activity_001",
                    "mock": True,
                },
            )
        try:
            import requests
            resp = requests.post(
                f"{self._api_url}/activities",
                json={
                    "contact_email": contact_email,
                    "type": activity_type,
                    "description": description,
                },
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            return ServiceResponse(success=True, data=resp.json())
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))
