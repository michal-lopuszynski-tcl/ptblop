import argparse
import pathlib
import json


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", required=True, type=pathlib.Path)
    parser.add_argument("-o", required=True, type=pathlib.Path)

    return parser.parse_args()


def convert_qwen32b_bpc_to_old(d):
    assert len(d) == 64

    res = {"attention_indices": 64 * [-1], "mlp_indices": 64 * [-1]}

    for k, v in d.items():
        i = int(k.split(".")[2])
        res["attention_indices"][i] = int(v["use_attention"])
        res["mlp_indices"][i] = int(v["use_mlp"])

    for i in range(64):
        assert res["attention_indices"][i] in {0, 1}
        assert res["mlp_indices"][i] in {0, 1}

    return res


def main(inp_fname, out_fname):
    if out_fname.exists():
        print(f"{out_fname} exists, aborting conversion")
    else:
        with open(inp_fname, "rt") as f:
            d = json.load(f)
        d_old = convert_qwen32b_bpc_to_old(d)
        with open(out_fname, "wt") as f:
            f.write(json.dumps(d_old) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args.i, args.o)
