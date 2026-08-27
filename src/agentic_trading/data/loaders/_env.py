# _env.py

"""Shared environment/secrets access for every loader in this package.

Loads `.env` once at import time via python-dotenv — the only place in this
project that reads secrets from disk — so every loader module can just read
`os.environ` via `require_env`. Never validates a present value's content,
only that it exists.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def require_env(var_name: str) -> str:
    """Returns the environment variable `var_name`, raising a clear,
    actionable error naming it if it's missing — never a bare KeyError."""
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {var_name!r}. Set it in .env or the "
            "environment before using this data loader."
        )
    return value
