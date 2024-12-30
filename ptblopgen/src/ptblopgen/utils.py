import datetime
import random
import string

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

    now_str = f"{now:%Y-%m-%d-%H%M-%S}{hundredths:02d}"
    return now_str


def get_random_str(n: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def get_versions() -> tuple[str, str]:
    return ptblop.__version__, _version.__version__


def get_bp_config_signature(bp_config):
    singature_strs = []

    for v in bp_config.values():
        v_signature_str = str(int(not v["use_attention"])) + str(int(not v["use_mlp"]))
        singature_strs.append(v_signature_str)
    signature_str = "".join(singature_strs)
    return int(signature_str, 2)
