from typing import Any

import argparse
import pathlib
import subprocess
import sys

import yaml

from ptblop import __version__ as ptblop_version
from .._version import __version__ as ptblopgen_version

REPRO_SUBDIR = "repro"


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
            if not line.startswith(
                "ptblop_version:"
            ) and not line.startswith("ptblopgen_version:"):
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


def main(args: argparse.Namespace):
    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    copy_config(args.config, output_path)
    save_requirements(
        output_path / REPRO_SUBDIR / "requirements.txt",
        output_path / REPRO_SUBDIR / "requirements_unsafe.txt",
    )
    config = read_config(args.config)

    print("Generating!!!")