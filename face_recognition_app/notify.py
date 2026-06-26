"""Email notifications for the door system.

Adapted from the sibling BrewAutomation project's notify.py: same Gmail
SMTP_SSL transport, same `.env` variable names (SENDER_EMAIL,
SENDER_APP_PASSWORD, RECIPIENT_EMAIL), and the same 0/1/2 exit-code contract —
extended here to attach images (the per-person snapshots).

Credentials are read from this repo's `.env` if present, otherwise from
BrewAutomation's `.env`, so the existing Gmail app password is reused without
copying secrets around. A real env var still overrides either file.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ENV = Path(__file__).resolve().parent.parent / ".env"
_BREW_ENV = Path.home() / "IdeaProjects" / "BrewAutomation" / ".env"


def load_env(paths=(_REPO_ENV, _BREW_ENV)):
    """Load key=value pairs from the first existing .env file(s) into the
    environment (without overriding real env vars)."""
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(
                        key.strip(), value.strip().strip('"').strip("'")
                    )


def _resolve_creds(sender, password, to):
    if not (sender and password and to):
        load_env()
        sender = sender or os.environ.get("SENDER_EMAIL", "")
        password = password or os.environ.get("SENDER_APP_PASSWORD", "")
        to = to or os.environ.get("RECIPIENT_EMAIL", "")
    return sender, password, to


def send(subject, body, attachments=(), sender="", password="", to=""):
    """Send an email via Gmail SMTP with optional image attachments.

    Returns 0 on success, 1 on credential/config error, 2 on SMTP/network error.
    """
    sender, password, to = _resolve_creds(sender, password, to)
    if not all([sender, password, to]):
        logger.error(
            "Missing email credentials (SENDER_EMAIL, SENDER_APP_PASSWORD, "
            "RECIPIENT_EMAIL). Set them in .env."
        )
        return 1

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"Door Monitor <{sender}>"
    msg["To"] = to
    msg.set_content(body)

    for path in attachments:
        path = Path(path)
        if not path.is_file():
            logger.warning("Attachment missing, skipping: %s", path)
            continue
        ctype, _ = mimetypes.guess_type(path.name)
        maintype, _, subtype = (ctype or "application/octet-stream").partition("/")
        msg.add_attachment(
            path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name
        )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        logger.info("Email sent to %s", to)
        return 0
    except smtplib.SMTPAuthenticationError:
        logger.error("Gmail authentication failed. Verify SENDER_APP_PASSWORD.")
        return 1
    except smtplib.SMTPException as exc:
        logger.error("SMTP failed (%s). May retry on the next event.", type(exc).__name__)
        return 2
    except OSError:
        logger.error("Network error sending email. Check the connection.")
        return 2


def send_event_email(event, sender="", password="", to=""):
    """Email one DoorEvent: subject summary, body with per-person detail, and
    the full frame + one crop per person attached."""
    subject = f"Door: {event.summary}"
    lines = [
        f"Detected at {event.timestamp}.",
        "",
        event.summary,
        "",
    ]
    for i, person in enumerate(event.people, 1):
        who = person.get("name") or person["label"].replace("_", " ")
        reasons = ", ".join(person.get("reasons", []))
        lines.append(f"  {i}. {who}" + (f" — {reasons}" if reasons else ""))
    lines += ["", "Full frame and per-person images attached."]

    attachments = [event.frame_path, *event.person_paths]
    return send(subject, "\n".join(lines), attachments=attachments,
                sender=sender, password=password, to=to)
