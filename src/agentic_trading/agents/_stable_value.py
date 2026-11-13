# _stable_value.py

"""A deterministic JSON description of an agent's parameters.

`stable_value` turns a value into JSON values that are the same for two
identically built values in any process: no memory address, no `repr`, and
no set or hash order enters it."""

import functools
import json
import types
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import BaseModel


def qualified_name(cls: type) -> str:
    """`cls` as `<module>.<qualname>`."""
    return f"{cls.__module__}.{cls.__qualname__}"


def stable_value(value: Any) -> Any:
    """`value` as JSON values, deterministically:

    - `None`, booleans, numbers and strings as themselves; enums by value;
      dates, datetimes and times in ISO 8601; a timedelta as its seconds;
      decimals, paths and UUIDs as strings; numpy scalars and arrays as
      their Python values;
    - lists and tuples as lists; sets as lists sorted by their canonical
      JSON; mappings as objects, a non-string key written as its canonical
      JSON;
    - a pydantic model as `{"class", "fields"}`, every field described
      recursively;
    - a class as `{"class"}`; a function as `{"callable"}`, a
      `functools.partial` with its arguments;
    - any other object as `{"class", "attributes"}`, its public instance
      attributes described recursively;
    - an object met again inside itself as `{"cycle"}`, its class.

    Raises `TypeError` for an object with none of these shapes."""
    return _stable(value, frozenset())


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _stable(value: Any, enclosing: frozenset[int]) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Enum):
        return _stable(value.value, enclosing)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Decimal | PurePath | UUID):
        return str(value)
    if isinstance(value, np.generic | np.ndarray):
        return _stable(value.tolist(), enclosing)
    if isinstance(value, type):
        return {"class": qualified_name(value)}
    if isinstance(value, types.FunctionType | types.BuiltinFunctionType | types.MethodType):
        return {"callable": f"{value.__module__}.{value.__qualname__}"}
    if id(value) in enclosing:
        return {"cycle": qualified_name(type(value))}
    inner = enclosing | {id(value)}
    if isinstance(value, list | tuple):
        return [_stable(item, inner) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_stable(item, inner) for item in value), key=_canonical)
    if isinstance(value, Mapping):
        described: dict[str, Any] = {}
        for key, item in value.items():
            stable_key = _stable(key, inner)
            name = stable_key if isinstance(stable_key, str) else _canonical(stable_key)
            described[name] = _stable(item, inner)
        return described
    if isinstance(value, functools.partial):
        return {
            "callable": _stable(value.func, inner),
            "args": _stable(value.args, inner),
            "keywords": _stable(value.keywords, inner),
        }
    if isinstance(value, BaseModel):
        return {
            "class": qualified_name(type(value)),
            "fields": {
                name: _stable(getattr(value, name), inner) for name in type(value).model_fields
            },
        }
    if hasattr(value, "__dict__"):
        return {
            "class": qualified_name(type(value)),
            "attributes": {
                name: _stable(item, inner)
                for name, item in vars(value).items()
                if not name.startswith("_")
            },
        }
    raise TypeError(f"no stable description of a {qualified_name(type(value))}")
