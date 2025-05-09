import argparse
import gzip
import logging
import pathlib
import platform
import shutil
import subprocess
import sys
from typing import Any, Optional

import yaml

from .. import modelgen, utils

REPRO_SUBDIR_PREFIX = "repro"
BP_CONFIG_SUBDIR_PREFIX = "bp_configs"

logger = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    subparser = subparsers.add_parser("sample")
    subparser.add_argument("--config", type=pathlib.Path, required=True)
    subparser.add_argument("--output-path", type=pathlib.Path, required=True)

    subparser = subparsers.add_parser("paretofind")
    subparser.add_argument("--config", type=pathlib.Path, required=True)
    subparser.add_argument("--output-path", type=pathlib.Path, required=True)
    subparser.add_argument(
        "--bp-configs-path", action="append", type=pathlib.Path, required=True
    )

    subparser = subparsers.add_parser("paretoeval")
    subparser.add_argument("--config", type=pathlib.Path, required=True)
    subparser.add_argument("--pareto-path", type=pathlib.Path, required=True)
    subparser.add_argument("--min-metric", type=float, default=None)
    subparser.add_argument("--min-mparams", type=float, default=None)
    subparser.add_argument("--max-mparams", type=float, default=None)
    subparser.add_argument("--pareto-level", type=int, default=None)
    subparser.add_argument("--no-shuffle", action="store_true")

    help_msg = parser.format_help()
    return parser.parse_args(), help_msg


def print_versions() -> None:
    v_ptblop, v_ptblopgen = utils.get_versions()
    print(f"ptblop version: {v_ptblop}")
    print(f"ptblopgen version: {v_ptblopgen}")


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
        "ptblopgen",
    ]:
        logging.getLogger(module_name).setLevel(logging.INFO)


def read_config(fname: str) -> dict[str, Any]:
    with open(fname, "rt") as f:
        return yaml.safe_load(f)


def copy_config(config_path: pathlib.Path, repro_path: pathlib.Path) -> None:
    v_ptblop, v_ptblopgen = utils.get_versions()
    config_copy_path = repro_path / "config.yaml"
    if config_copy_path.exists():
        msg = f"Config copy already exists, please delete it first, {config_copy_path}"
        raise FileExistsError(msg)
    config_copy_path.parent.mkdir(exist_ok=True, parents=True)
    with open(config_path, "rt") as f_in, open(config_copy_path, "wt") as f_out:
        f_out.write(f'ptblop_version: "{v_ptblop}"\n')
        f_out.write(f'ptblopgen_version: "{v_ptblopgen}"\n\n')
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


def make_repro_dir(
    args: argparse.Namespace,
    repro_subdir_prefix: str,
    bp_configs_paths: Optional[list[pathlib.Path]] = None,
) -> None:
    repro_subdir = repro_subdir_prefix + "." + utils.get_timestamp_for_fname()
    repro_path = args.output_path / repro_subdir
    copy_config(args.config, repro_path)
    save_requirements(
        repro_path / "requirements.txt", repro_path / "requirements_unsafe.txt"
    )

    if bp_configs_paths is None:
        bp_configs_paths = [args.output_path / modelgen.BP_CONFIG_DB_FNAME]

    for i, cur_bp_configs_path in enumerate(bp_configs_paths, start=1):
        if cur_bp_configs_path.exists():
            cur_dir = repro_path / f"{BP_CONFIG_SUBDIR_PREFIX}_{i:02d}"
            cur_dir.mkdir(parents=True, exist_ok=True)
            bp_config_bak_path = cur_dir / (modelgen.BP_CONFIG_DB_FNAME + ".gz")

            with open(cur_bp_configs_path, "rb") as f_in:
                with gzip.open(bp_config_bak_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)


def main() -> int:
    setup_logging()
    args, help_msg = parse_args()
    if args.version:
        print_versions()
    else:
        logger.info(f"Running on node {platform.node()}")
        if args.command == "sample":
            args.output_path.mkdir(exist_ok=True, parents=True)
            make_repro_dir(args, REPRO_SUBDIR_PREFIX)
            config = read_config(args.config)
            modelgen.main_sample(config, args.output_path)
        elif args.command == "paretofind":
            args.output_path.mkdir(exist_ok=True, parents=True)
            make_repro_dir(
                args, REPRO_SUBDIR_PREFIX, bp_configs_paths=args.bp_configs_path
            )
            config = read_config(args.config)
            modelgen.main_paretofind(
                config=config,
                output_path=args.output_path,
                bp_config_db_paths=args.bp_configs_path,
            )
        elif args.command == "paretoeval":
            config = read_config(args.config)
            modelgen.main_paretoeval(
                config=config,
                pareto_path=args.pareto_path,
                min_metric=args.min_metric,
                shuffle=not args.no_shuffle,
                min_mparams=args.min_mparams,
                max_mparams=args.max_mparams,
                pareto_level=args.pareto_level,
            )
        else:
            if args.command is None:
                print("No command given\n")
            else:
                print(f"Unknown command {args.command}\n")
            print(help_msg)
