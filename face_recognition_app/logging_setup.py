"""Shared logging setup for CLI entrypoints."""

from __future__ import annotations

import logging


def configure(verbose=False):
    """Configure root logging for a CLI run.

    INFO by default; DEBUG when verbose is True. Format is concise so
    one-line status messages don't get drowned in metadata.

    force=True is required because hashlib (and other stdlib modules)
    may call logging.error() at import time, which silently installs a
    default handler. Without force=True our basicConfig is a no-op and
    INFO logs stay invisible.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
