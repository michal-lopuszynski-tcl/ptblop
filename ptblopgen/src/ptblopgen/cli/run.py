import argparse
import logging
import pathlib
import subprocess
import sys
from typing import Any

import yaml
from ptblop import __version__ as ptblop_version

from .. import modelgen
from .._version import __version__ as ptblopgen_version


REPRO_SUBDIR = "repro"


def parse_args() -> tuple[argparse.Namespace, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    parser_gen = subparsers.add_parser("gen")
    parser_gen.add_argument("--output-path", type=pathlib.Path, required=True)
    parser_gen.add_argument("--config", type=pathlib.Path, required=True)
    help_msg = parser.format_help()

    return parser.parse_args(), help_msg


def print_versions() -> None:
    print(f"ptblop version: {ptblop_version}")
    print(f"ptblopgen version: {ptblopgen_version}")


def setup_logging() -> None:
    fmt = (
        "%(asctime)s.%(msecs)03d500: %(levelname).1s "
        + "%(name)s.py:%(lineno)d] %(message)s"
    )
    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Here you put modules where you want more verbose logging

    for module_name in [
        __name__,
        "ptblop",
        "modelgen",
        "builders",
    ]:
        logging.getLogger(module_name).setLevel(logging.INFO)


def read_config(fname: str) -> dict[str, Any]:
    with open(fname, "rt") as f:
        return yaml.safe_load(f)


def copy_config(config_path: pathlib.Path, output_path: pathlib.Path) -> None:
    config_copy_path = output_path / REPRO_SUBDIR / "config.yaml"
    if config_copy_path.exists():
        msg = f"Config copy already exists, please delete it first, {config_copy_path}"
        raise FileExistsError(msg)
    config_copy_path.parent.mkdir(exist_ok=True, parents=True)
    with open(config_path, "rt") as f_in, open(config_copy_path, "wt") as f_out:
        f_out.write(f'ptblop_version: "{ptblop_version}"\n')
        f_out.write(f'ptblopgen_version: "{ptblopgen_version}"\n\n')
        for line in f_in:
            if not line.startswith("ptblop_version:") and not line.startswith(
                "ptblopgen_version:"
            ):
                f_out.write(f"{line}")


def save_requirements(
    requirements_path: pathlib.Path, requirements_unsafe_path: pathlib.Path
) -> None:
    # Dump "normal" requirements

    result = subprocess.run(
        [sys.executable, "-mpip", "freeze"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    requirements_safe = result.stdout.decode("utf-8").splitlines()

    with requirements_path.open("wt") as f:
        f.write(f"# Python {sys.version}\n\n")
        for r in requirements_safe:
            f.write(r + "\n")

    # Dump "unsafe" requirements (rarely needed)

    result = subprocess.run(
        [sys.executable, "-mpip", "freeze", "--all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    requirements_all = result.stdout.decode("utf-8").splitlines()

    with requirements_unsafe_path.open("wt") as f:
        f.write(f"# Python {sys.version}\n\n")
        for r in requirements_all:
            if r not in requirements_safe:
                f.write(r + "\n")


def main() -> int:
    setup_logging()
    args, help_msg = parse_args()
    if args.version:
        print_versions()
    else:
        if args.command == "gen":
            output_path = pathlib.Path(args.output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            copy_config(args.config, output_path)
            save_requirements(
                output_path / REPRO_SUBDIR / "requirements.txt",
                output_path / REPRO_SUBDIR / "requirements_unsafe.txt",
            )
            config = read_config(args.config)
            modelgen.main_sample_random(config, output_path)
        else:
            if args.command is None:
                print("No command given\n")
            else:
                print(f"Unknown command {args.command}\n")
            print(help_msg)
