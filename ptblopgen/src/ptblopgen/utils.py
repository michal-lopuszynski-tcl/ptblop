import datetime
import json
import pathlib
import random
import string
from typing import Any

import ptblop

try:
    import awq

    WQLINEAR_GEMM_TYPE = awq.modules.linear.gemm.WQLinear_GEMM
except ImportError:
    WQLINEAR_GEMM_TYPE = type(None)

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


def _get_weight_size_dict(model, prefix, fill_none):
    if prefix:
        prefix = f"{prefix}."

    if isinstance(model, WQLINEAR_GEMM_TYPE):
        res = {}
        for k in model.state_dict():
            if not fill_none:
                if k == "qweight":
                    res[f"{prefix}qweight"] = (
                        model.in_features * model.out_features,
                        model.qweight.data_ptr(),
                    )
                elif k == "bias":
                    res[f"{prefix}bias"] = (model.bias.numel(), model.bias.data_ptr())
                else:
                    res[f"{prefix}{k}"] = (None, None)
            else:
                res[f"{prefix}{k}"] = (None, None)
        return res
    else:
        if isinstance(model, ptblop.PrunableBlock):
            skip_names = model.get_unused_layer_names()
        else:
            skip_names = set()

        res = {}

        for name, submodel in model.named_children():
            if name not in skip_names:
                res_cur = _get_weight_size_dict(submodel, f"{prefix}{name}", fill_none)
            else:
                res_cur = _get_weight_size_dict(submodel, f"{prefix}{name}", True)
            res |= res_cur

        sd = model.state_dict()

        for k in sd:
            kk = f"{prefix}{k}"
            if kk not in res:
                if fill_none:
                    res[kk] = (None, None)
                else:
                    v = sd[k]
                    res[kk] = (v.numel(), v.data_ptr())
        return res


def get_num_active_params(model):
    # This function improves handling of AWQ quantized models.
    # For the AWQ model it returns the same number of parameter as for bfloat16 model
    # Note. This is not always desireable, perhas you want to prune towards model byte
    # size.
    wd_raw = _get_weight_size_dict(model, "", False)
    wd = {k: v for k, v in wd_raw.items() if v[0] is not None}
    ptrs = {v[1]: v[0] for v in wd.values()}
    return sum(ptrs.values())
