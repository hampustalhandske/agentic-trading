# lake_errors.py

"""`LakeReadError`, the base class of every error the data layer raises
when a lake read itself failed.

A run must stop on such an error rather than record it as an analyst's or
a manager's failure: the signal would rest on data the run could not read,
not on a model that failed on data it had. Every place that turns an
agent's exception into a flat error signal re-raises a `LakeReadError`
first. It derives from `RuntimeError`, so a caller catching
`RuntimeError` still sees every one of them."""


class LakeReadError(RuntimeError):
    """A lake read failed: the lake refused it or could not be reached, its
    answer was cut off, it could not be pinned to the session's lake time
    or version, or it was asked for data past the session's cut."""
