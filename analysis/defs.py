
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