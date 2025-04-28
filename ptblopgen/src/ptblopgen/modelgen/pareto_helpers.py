import enum
import logging

import numpy as np


class Mode(enum.Enum):
    O1_MAX_O2_MAX = 1
    O1_MAX_O2_MIN = 2
    O1_MIN_O2_MAX = 3
    O1_MIN_O2_MIN = 4


logger = logging.getLogger(__name__)


def get_dedpuplicated_copy(o1, o2):
    ii = np.lexsort((o2, o1))
    o1_sorted = o1[ii]
    o2_sorted = o2[ii]
    delta_o1 = o1_sorted[1:] - o1_sorted[:-1]
    delta_o2 = o2_sorted[1:] - o2_sorted[:-1]
    non_duplicates = (delta_o1 != 0) | (delta_o2 != 0)
    nn = len(o1) - np.sum(non_duplicates) - 1
    logger.info(f"Removing {nn} duplicates")
    non_duplicates = non_duplicates.tolist()
    non_duplicates.append(True)
    non_duplicates = np.array(non_duplicates)
    return o1[ii[non_duplicates]], o2[ii[non_duplicates]]


def _get_pf_mask_max_max(o1, o2):
    # o1, o2 -> maximized
    # assert not _has_duplicates(o1, o2)
    o1_, o2_ = get_dedpuplicated_copy(o1, o2)
    n = len(o1_)
    pareto_pairs = set()

    for i in range(n):
        o1i = o1_[i]
        o2i = o2_[i]
        is_pareto = np.sum((o1_ >= o1i) & (o2_ >= o2i)) <= 1
        if is_pareto:
            pareto_pairs.add((o1i, o2i))

    return np.fromiter((pair in pareto_pairs for pair in zip(o1, o2)), dtype=bool)


def get_pf_mask(o1, o2, mode):
    if mode == Mode.O1_MAX_O2_MAX:
        return _get_pf_mask_max_max(o1, o2)
    elif mode == Mode.O1_MAX_O2_MIN:
        return _get_pf_mask_max_max(o1, -o2)
    elif mode == Mode.O1_MIN_O2_MAX:
        return _get_pf_mask_max_max(-o1, o2)
    elif mode == Mode.O1_MIN_O2_MIN:
        return _get_pf_mask_max_max(-o1, -o2)
    else:
        raise ValueError(f"Unknown {mode=}")
