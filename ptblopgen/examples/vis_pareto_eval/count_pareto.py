import argparse
import json
import pathlib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto-path", "-p", type=pathlib.Path)
    parser.add_argument("--min-mparams", "-n", type=float)
    parser.add_argument("--max-mparams", "-x", type=float)
    return parser.parse_args()


def main(args):
    with open(args.pareto_path, "rt") as f:
        n = 0
        n_matching = 0
        for line in f:
            d = json.loads(line)
            if (
                d["mparams_pred"] >= args.min_mparams
                and d["mparams_pred"] <= args.max_mparams
            ):
                n_matching += 1
            n += 1
    print(f"{n=} {n_matching=}")


if __name__ == "__main__":
    main(parse_args())
