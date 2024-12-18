import datetime

import ptblop

from . import _version


def get_timestamp() -> str:
    current_utc = datetime.datetime.now(datetime.timezone.utc)
    return current_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_timestamp_for_fname() -> str:
    current_utc = datetime.datetime.now(datetime.timezone.utc)
    return current_utc.strftime("%Y-%m-%d_%H%M%S")


def get_versions() -> tuple[str, str]:
    return ptblop.__version__, _version.__version__
