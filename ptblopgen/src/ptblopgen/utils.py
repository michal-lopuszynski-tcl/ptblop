import datetime
import json
import pathlib
import random
import string
from typing import Any

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


def make_runid():
    return f"{get_timestamp_for_fname()}_{get_random_str(6)}"


def get_versions() -> tuple[str, str]:
    return ptblop.__version__, _version.__version__


def get_bp_config_signature(bp_config):
    singature_strs = []

    for v in bp_config.values():
        v_signature_str = str(int(not v["use_attention"])) + str(int(not v["use_mlp"]))
        singature_strs.append(v_signature_str)
    signature_str = "".join(singature_strs)
    return int(signature_str, 2)


def bp_config_signature_to_str(bp_config_signature: int) -> str:
    return hex(bp_config_signature)[2:]


def bp_config_signature_from_str(s: str) -> int:
    return int(s, 16)


def get_bp_config_signature_str(bp_config):
    return bp_config_signature_to_str(get_bp_config_signature(bp_config))


def update_db(
    db_path: pathlib.Path, db_entry: dict[str, Any], mode: str = "append"
) -> None:
    if mode == "append":
        flag = "at"
    elif mode == "reset":
        flag = "wt"
    else:
        raise ValueError(f"Unknown mode {mode}")

    with open(db_path, flag) as f:
        f.write(json.dumps(db_entry) + "\n")
