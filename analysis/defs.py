
import re
from collections import namedtuple

Stats = namedtuple('Stats', ['avg', 'std'])

# Integer codes for the global-timeline 'event_type' column.
# Stored as int8 instead of Python strings so the timeline stays a compact,
# purely-numeric frame: melting to strings made event_type an object column
# (one boxed Python object per row) and forced the whole array to object dtype
# in LogAnalyzer's overtake scan, which was a major memory sink.
INVOCATION = 0
ACQUISITION = 1
RELEASE = 2

# Maps melt's var_name (the original timestamp column name) to its int code.
EVENT_CODES = {
    'invocation': INVOCATION,
    'acquisition': ACQUISITION,
    'release': RELEASE,
}

# Run directories are named <lock>_<threads>_<pin>_w<work>, written by
# scripts/runner.py:run_dir_id(). This is the single reader of that name.
# The thread count it carries is the only authoritative record of how many
# threads a run spawned: a thread that completed no operations leaves no rows
# in the parquet, so counting distinct thread ids in the data silently drops it
# -- and dropping a starved thread makes an unfair lock score as perfectly fair.
#
# Anchored, because an unanchored match would accept the `<lock>_<threads>_<pin>`
# prefix of a work-suffixed directory and silently collapse every work size onto
# one line. The work group is optional so directories from sweeps predating the
# work dimension still parse.
RUN_DIR_PATTERN = re.compile(
    r'^(?P<lock>[A-Za-z]+)_(?P<threads>\d+)_(?P<pin>\d+)(?:_w(?P<work>\d+))?$'
)


def parse_run_dir_id(folder_name: str) -> dict | None:
    """Parse `mcs_8_1_w10000` (or legacy `mcs_8_1`) into its parameters.

    Returns None when the name doesn't match, so callers can fall back instead
    of crashing on a hand-made or legacy directory."""
    match = RUN_DIR_PATTERN.match(folder_name)
    if match is None:
        return None
    work = match.group('work')
    return {
        'lock': match.group('lock'),
        'threads': int(match.group('threads')),
        'pin': int(match.group('pin')),
        'work': int(work) if work is not None else None,
    }
