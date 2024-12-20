import datetime

import ptblop

from . import _version


def get_timestamp() -> str:
    current_utc = datetime.datetime.now(datetime.timezone.utc)
    return current_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_timestamp_for_fname() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)

    # Round to nearest hundredth of a second
    hundredths = round(now.microsecond / 10000)

    # Handle the case where rounding results in 100
    if hundredths == 100:
        now = now + datetime.timedelta(seconds=1)
        hundredths = 0

    now_str = f"{now:%Y-%m-%d_%H%M-%S}{hundredths:02d}"
    return now_str


def get_versions() -> tuple[str, str]:
    return ptblop.__version__, _version.__version__
