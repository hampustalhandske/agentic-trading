# lake_answer_cache.py

"""An on-disk store of lake `/v1/sql` answers read inside a `lake_session`,
so a run that restarts or resumes reads each answer it already has from
disk instead of the lake.

A pinned read never changes its answer: the lake serves the same rows for
the same SQL and parameters at the same lake time T from the same lake
code. An answer is therefore stored under `answer_key`, the sha256 of the
canonical JSON of the store's `FORMAT`, the session's lake version (the
`<code>.<builds>` pair it pins or recorded), T, the exact SQL text and the
parameters. A key never names an answer of another version, lake time or
SQL.

Layout: `<directory>/<key[:2]>/<key>.parquet`, one file per answer,
holding the Arrow table exactly as the lake sent it (types, time zones and
schema metadata). The file's schema metadata also holds the key's inputs
and the Arrow schema of the answer; a read checks both and restores the
schema, so a file read back equals the answer that was stored. A file is
written to a temporary name beside it and renamed into place, so a kill
mid-write never leaves a partial entry under a key. An entry that cannot
be read, or whose recorded inputs do not match its key, is removed and
logged, and the caller fetches the answer again."""

import base64
import hashlib
import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel

from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

# The layout of a stored answer; a change of layout changes every key.
FORMAT = 2
# Schema-metadata keys of a stored file: the key's inputs, and the answer's Arrow schema.
_INPUTS_FIELD = b"agentic_lake_answer.inputs"
_SCHEMA_FIELD = b"agentic_lake_answer.schema"


class LakeAnswerCacheCounts(BaseModel):
    """What a `LakeAnswerCache` did since it was opened: `hits` answers read
    from disk, `stored` answers fetched from the lake and written,
    `unstored` answers fetched and not written (no lake version known, or
    an empty answer the caller asked not to keep), and `discarded` entries
    removed because they could not be read or did not match their key."""

    hits: int = 0
    stored: int = 0
    unstored: int = 0
    discarded: int = 0


def _canonical(value: Any) -> str:
    """`value` as canonical JSON: sorted keys, no whitespace. A value JSON
    cannot hold (a date, a timestamp) is written as its type name and its
    `str`, so two values of different types never share a key."""

    def tagged(item: Any) -> Any:
        return {"type": type(item).__qualname__, "value": str(item)}

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=tagged)


def answer_inputs(lake_version: str, lake_time: datetime, query: str, params: list[Any]) -> str:
    """The canonical JSON of everything an answer depends on: `FORMAT`,
    `lake_version`, `lake_time` (as UTC ISO 8601), the exact `query` text
    and `params` in order."""
    return _canonical(
        {
            "format": FORMAT,
            "lake_version": lake_version,
            "lake_time": lake_time.astimezone(UTC).isoformat(),
            "sql": query,
            "params": params,
        }
    )


def answer_key(inputs: str) -> str:
    """The sha256 hex digest of `inputs` (`answer_inputs`)."""
    return hashlib.sha256(inputs.encode()).hexdigest()


class LakeAnswerCache:
    """The answers stored under `directory` (created on first write). Safe
    to share across threads; each method touches one entry."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._counts = LakeAnswerCacheCounts()

    def counts(self) -> LakeAnswerCacheCounts:
        """A copy of what the cache has done since it was opened."""
        with self._lock:
            return self._counts.model_copy()

    def _count(self, field: str) -> None:
        with self._lock:
            setattr(self._counts, field, getattr(self._counts, field) + 1)

    def count_unstored(self) -> None:
        """Counts an answer fetched from the lake and not stored."""
        self._count("unstored")

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.parquet"

    def get(self, inputs: str) -> pa.Table | None:
        """The answer stored for `inputs` (`answer_inputs`), `None` when
        there is none. An entry that cannot be read back whole, or that
        records other inputs, is removed, logged and gives `None`."""
        path = self._path(answer_key(inputs))
        if not path.exists():
            return None
        try:
            stored = pq.read_table(path)
            metadata = dict(stored.schema.metadata or {})
            recorded = metadata.pop(_INPUTS_FIELD).decode()
            schema = pa.ipc.read_schema(pa.py_buffer(base64.b64decode(metadata.pop(_SCHEMA_FIELD))))
            if recorded != inputs:
                raise ValueError("the entry records other inputs than its key's")
            table = stored.replace_schema_metadata(metadata or None)
            if not table.schema.equals(schema, check_metadata=True):
                table = table.cast(schema)
            if not table.schema.equals(schema, check_metadata=True):
                raise ValueError("the entry's schema does not match the answer's")
        except Exception as error:
            path.unlink(missing_ok=True)
            self._count("discarded")
            logger.warning(
                "Discarded the lake answer cache entry %s (%s: %s); the answer is read from "
                "the lake again",
                path,
                type(error).__name__,
                error,
            )
            return None
        self._count("hits")
        return table

    def put(self, inputs: str, table: pa.Table) -> None:
        """Stores `table` as the answer for `inputs` (`answer_inputs`):
        written to a temporary file beside its path, then renamed over
        it. A write that fails (a full disk) is logged and counted as
        unstored; the answer is still the caller's."""
        path = self._path(answer_key(inputs))
        metadata = dict(table.schema.metadata or {})
        metadata[_INPUTS_FIELD] = inputs.encode()
        metadata[_SCHEMA_FIELD] = base64.b64encode(table.schema.serialize().to_pybytes())
        partial = path.with_name(f".{path.name}.{os.getpid()}-{uuid.uuid4().hex}.partial")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                table.replace_schema_metadata(metadata),
                partial,
                use_compliant_nested_type=False,
            )
            os.replace(partial, path)
        except (OSError, pa.ArrowException) as error:
            self._count("unstored")
            logger.warning(
                "Could not store a lake answer at %s (%s: %s)", path, type(error).__name__, error
            )
            return
        finally:
            partial.unlink(missing_ok=True)
        self._count("stored")
