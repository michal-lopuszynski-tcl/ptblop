#!/usr/bin/env python3

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--version-file", required=True)
    parser.add_argument("--version-segment", type=int, required=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    lines = []

    with open(args.version_file, "rt") as f:
        for line in f:
            if line.strip().startswith("__version__ ="):
                line_segments = line.split("=")
                assert len(line_segments) == 2
                version_segments = line_segments[1].strip()[1:-1].split(".")
                version_new = int(version_segments[args.version_segment]) + 1
                version_segments[args.version_segment] = str(version_new)
                line_new = line_segments[0] + '= "' + ".".join(version_segments) + '"\n'
                lines.append(line_new)
            else:
                lines.append(line)

    with open(args.version_file, "wt") as f:
        for line in lines:
            f.write(line)


if __name__ == "__main__":
    main(parse_args())
