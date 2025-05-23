#!/usr/bin/env python3

import argparse
import json
import pathlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-file", default=pathlib.Path("./results.json"), type=pathlib.Path
    )
    return parser.parse_args()


def main(args):
    with open(args.results_file) as f:
        d = json.load(f)
    solutions_all = d["eval"]

    solutions_id = sorted(solutions_all.keys())

    for sol_id in solutions_id:
        solutions_for_id = solutions_all[sol_id]
        for j, sol in enumerate(solutions_for_id, start=1):
            print(f"{sol_id}.{j} OUTPUTS RAW:\n")
            for o in sol["outputs_raw"]:
                print(o)
            print(f"{sol_id}.{j} SOLUTION\n")
            print(sol["solution"])


if __name__ == "__main__":
    main(parse_args())
