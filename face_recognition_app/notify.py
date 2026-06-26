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
import time
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


def send(subject, text_body, html_body="", inline_images=(), attachments=(),
         sender="", password="", to=""):
    """Send an email via Gmail SMTP.

    text_body: plain-text fallback. html_body: optional rich HTML. inline_images:
    iterable of (cid, path) embedded in the HTML via cid: references.
    attachments: iterable of file paths attached normally.

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
    msg.set_content(text_body)

    if html_body:
        msg.add_alternative(html_body, subtype="html")
        html_part = msg.get_payload()[-1]
        for cid, path in inline_images:
            path = Path(path)
            if not path.is_file():
                continue
            html_part.add_related(
                path.read_bytes(), maintype="image", subtype="jpeg", cid=f"<{cid}>"
            )

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


# Per-status presentation: (badge text, badge colour).
_BADGE = {
    "known": ("Known", "#2e7d32"),
    "unknown": ("Unknown", "#ef6c00"),
    "likely_delivery": ("Likely delivery", "#1565c0"),
}


def _friendly_time(iso_ts):
    """Turn '2026-06-26T19:20:55' into 'June 26, 2026 at 7:20 PM'."""
    try:
        dt = time.strptime(iso_ts, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError):
        return iso_ts
    hour = dt.tm_hour % 12 or 12
    ampm = "AM" if dt.tm_hour < 12 else "PM"
    return (f"{time.strftime('%B', dt)} {dt.tm_mday}, {dt.tm_year} "
            f"at {hour}:{dt.tm_min:02d} {ampm}")


def _headline(people):
    """A clear, scannable subject/headline for who is at the door."""
    labels = [p["label"] for p in people]
    known = [p["name"] for p in people if p["label"] == "known" and p.get("name")]
    unknown_n = labels.count("unknown")

    if "likely_delivery" in labels:
        return "📦 Possible delivery at your door"
    if known and not unknown_n:
        if len(known) == 1:
            return f"✅ {known[0]} is at your door"
        return f"✅ {', '.join(known)} are at your door"
    if known and unknown_n:
        return f"👀 {', '.join(known)} + {unknown_n} unknown at your door"
    if unknown_n == 1:
        return "⚠️ Unknown person at your door"
    if unknown_n > 1:
        return f"⚠️ {unknown_n} unknown people at your door"
    return "👤 Someone is at your door"


def _who(person):
    return person.get("name") or "Unknown"


def _clean_reasons(person):
    """Drop the redundant 'recognised X' note for known people."""
    return [r for r in person.get("reasons", []) if not r.startswith("recognised ")]


def _build_html(headline, friendly, people):
    cards = []
    for i, person in enumerate(people, 1):
        badge_text, color = _BADGE.get(person["label"], ("Person", "#555"))
        reasons = _clean_reasons(person)
        reason_html = (
            f'<div style="color:#777;font-size:13px;margin-top:6px;">'
            f'{"; ".join(reasons)}</div>' if reasons else ""
        )
        img_html = (
            f'<img src="cid:person{i}" width="120" '
            f'style="border-radius:8px;display:block;" alt="{_who(person)}">'
            if person.get("image") else ""
        )
        cards.append(
            '<table role="presentation" cellpadding="0" cellspacing="0" '
            'style="width:100%;border:1px solid #eee;border-radius:10px;'
            'margin-bottom:12px;"><tr>'
            f'<td style="padding:10px;width:140px;vertical-align:top;">{img_html}</td>'
            '<td style="padding:10px;vertical-align:top;">'
            f'<div style="font-size:17px;font-weight:600;color:#222;">{_who(person)}</div>'
            f'<span style="display:inline-block;margin-top:6px;padding:3px 10px;'
            f'border-radius:12px;background:{color};color:#fff;font-size:12px;">'
            f'{badge_text}</span>{reason_html}'
            '</td></tr></table>'
        )

    return (
        '<div style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,sans-serif;'
        'max-width:560px;margin:0 auto;color:#222;">'
        f'<h2 style="margin:0 0 2px;">{headline}</h2>'
        f'<p style="color:#888;margin:0 0 18px;font-size:14px;">{friendly}</p>'
        f'{"".join(cards)}'
        '<div style="margin-top:8px;">'
        '<div style="color:#888;font-size:13px;margin-bottom:6px;">Full view</div>'
        '<img src="cid:frame" style="width:100%;border-radius:10px;" alt="Full view">'
        '</div>'
        '<p style="color:#aaa;font-size:12px;margin-top:18px;">'
        'Sent by your Door Monitor.</p></div>'
    )


def send_event_email(event, sender="", password="", to=""):
    """Email one DoorEvent as an intuitive HTML message: a clear headline, a
    friendly time, and each person shown inline with a status badge."""
    people = event.people
    headline = _headline(people)
    friendly = _friendly_time(event.timestamp)

    # Plain-text fallback for clients that don't render HTML.
    text_lines = [headline, friendly, ""]
    for i, person in enumerate(people, 1):
        badge = _BADGE.get(person["label"], ("Person", ""))[0]
        reasons = _clean_reasons(person)
        line = f"{i}. {_who(person)} — {badge}"
        if reasons:
            line += f" ({'; '.join(reasons)})"
        text_lines.append(line)

    inline = [
        (f"person{i}", p["image"])
        for i, p in enumerate(people, 1)
        if p.get("image")
    ]
    if event.frame_path:
        inline.append(("frame", event.frame_path))

    html = _build_html(headline, friendly, people)
    return send(headline, "\n".join(text_lines), html_body=html,
                inline_images=inline, sender=sender, password=password, to=to)
