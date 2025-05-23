#!/usr/bin/env python3

import argparse
import json
import pathlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bp-configs-path", "-b", type=pathlib.Path, required=True)
    return parser.parse_args()


METRIC_THRESHOLD = 1.0e-5


def main(args):
    results = []
    with open(args.bp_configs_path) as f:
        for line in f:
            results.append(json.loads(line))

    if "mbpp" in results[0]["evaluation"]:
        base_metric_name = "mbpp"
    elif "huaneval" in results[0]["evaluation"]:
        base_metric_name = "humaneval"
    else:
        raise ValueError("Neither mbpp nor humaneval metric found")
    plus_metric_name = f"{base_metric_name}_plus"

    n_zeros_early = 0
    n_zeros_nonearly = 0
    n = len(results)

    for r in results:
        base_metric_value = r["evaluation"][base_metric_name]
        plus_metric_value = r["evaluation"][plus_metric_name]
        early_stopped = r["evaluation"]["early_stopped"]
        if (
            abs(base_metric_value) < METRIC_THRESHOLD
            and abs(plus_metric_value) < METRIC_THRESHOLD
        ):
            if early_stopped:
                n_zeros_early += 1
            else:
                n_zeros_nonearly += 1
    n_zeros = n_zeros_early +  n_zeros_nonearly
    f_zeros = n_zeros / n*100
    f_zeros_early = n_zeros_early / n * 100
    ff_zeros_eraly = n_zeros_early / n_zeros * 100
    print(f"{n:4d}         - total benchmarks")
    print(f"{n_zeros:4d} {f_zeros:.1f}%     - total zeros")
    print(f"{n_zeros_early:4d} {f_zeros_early:.1f}% {ff_zeros_eraly:.1f}% - total zeros from early stopping")


if __name__ == "__main__":
    main(parse_args())
