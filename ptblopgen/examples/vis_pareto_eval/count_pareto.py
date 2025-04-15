import argparse
import gzip
import json
import pathlib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto-path", "-p", required=True, type=pathlib.Path)
    parser.add_argument("--min-mparams", "-n", required=True, type=float)
    parser.add_argument("--max-mparams", "-x", required=True, type=float)
    return parser.parse_args()


def open_rt(fname):
    if str(fname).endswith(".gz"):
        return gzip.open(fname, "rt")
    else:
        return open(fname, "rt")


def main(args):
    with open_rt(args.pareto_path) as f:
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
