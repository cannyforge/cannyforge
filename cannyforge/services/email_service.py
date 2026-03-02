"""
CannyForge Email Service

Sends emails via SendGrid or AWS SES. Mock mode when no API key is set.

Usage:
    service = EmailService()
    service.connect()
    response = service.send_email(
        to="user@example.com",
        subject="Hello",
        body="Hello from CannyForge!"
    )
"""

import logging
import os
from cannyforge.services.service_base import ServiceResponse

logger = logging.getLogger("EmailService")


class EmailService:
    """Email integration with SendGrid/SES, mock mode when no API key."""

    def __init__(self):
        self._sendgrid_key = os.environ.get("SENDGRID_API_KEY")
        self._ses_region = os.environ.get("AWS_SES_REGION")
        self._mock = self._sendgrid_key is None and self._ses_region is None
        self._provider = None

    @property
    def is_mock(self) -> bool:
        return self._mock

    def connect(self):
        if self._mock:
            logger.info("Email: running in mock mode (set SENDGRID_API_KEY or AWS_SES_REGION for real)")
            return

        if self._sendgrid_key:
            self._provider = "sendgrid"
            logger.info("Email: using SendGrid")
        elif self._ses_region:
            self._provider = "ses"
            logger.info(f"Email: using AWS SES ({self._ses_region})")

    def send_email(self, to: str, subject: str, body: str,
                   from_email: str = "noreply@cannyforge.dev") -> ServiceResponse:
        if self._mock:
            return ServiceResponse(
                success=True,
                data={
                    "to": to, "subject": subject, "from": from_email,
                    "message_id": "mock_msg_001", "mock": True,
                },
            )

        if self._provider == "sendgrid":
            return self._send_sendgrid(to, subject, body, from_email)
        elif self._provider == "ses":
            return self._send_ses(to, subject, body, from_email)

        return ServiceResponse(success=False, error="No email provider configured")

    def _send_sendgrid(self, to, subject, body, from_email) -> ServiceResponse:
        try:
            import sendgrid
            from sendgrid.helpers.mail import Mail
            sg = sendgrid.SendGridAPIClient(api_key=self._sendgrid_key)
            message = Mail(from_email=from_email, to_emails=to,
                          subject=subject, plain_text_content=body)
            response = sg.send(message)
            return ServiceResponse(
                success=response.status_code in (200, 202),
                data={"status_code": response.status_code},
            )
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))

    def _send_ses(self, to, subject, body, from_email) -> ServiceResponse:
        try:
            import boto3
            client = boto3.client("ses", region_name=self._ses_region)
            response = client.send_email(
                Source=from_email,
                Destination={"ToAddresses": [to]},
                Message={
                    "Subject": {"Data": subject},
                    "Body": {"Text": {"Data": body}},
                },
            )
            return ServiceResponse(
                success=True,
                data={"message_id": response["MessageId"]},
            )
        except Exception as e:
            return ServiceResponse(success=False, error=str(e))
