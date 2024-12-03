import argparse
import pathlib
import logging

from . import run_gen


def parse_args() -> tuple[argparse.Namespace, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='store_true')
    subparsers = parser.add_subparsers(dest="command")

    parser_gen = subparsers.add_parser('gen')
    parser_gen.add_argument('--output-path', type=pathlib.Path, required=True)
    parser_gen.add_argument('--config', type=pathlib.Path, required=True)
    help_msg = parser.format_help()

    return parser.parse_args(), help_msg


def print_versions() -> None:
    from ptblop import __version__
    from .. import _version
    print(f"ptblop version: {__version__}")
    print(f"ptblopgen version: {_version.__version__}")


# def parse_args() -> argparse.Namespace:
#     # Try to parse --version

#     arg_parser = argparse.ArgumentParser()
#     subparsers = parser.add_subparsers(help='subcommand help')
#     # arg_parser.add_argument("--version", action="store_true")
#     # arg_parser.add_argument("--output-path", type=pathlib.Path)
#     # arg_parser.add_argument("--active-learning", type=int, default=0)
#     # # arg_parser.add_argument("--config", type=pathlib.Path)

#     # args = arg_parser.parse_args()

#     # # If no --version, run parsing of trainign/decomposition arguments

#     # if not args.version:
#     #     arg_parser = argparse.ArgumentParser()
#     #     arg_parser.add_argument(
#     #         "--output-path",
#     #         type=pathlib.Path,
#     #         required=True,
#     #     )
#     #     arg_parser.add_argument("--active-learning", type=int, default=0)
#     #     # arg_parser.add_argument(
#     #     #     "--config",
#     #     #     type=pathlib.Path,
#     #     #     required=True,
#     #     # )
#     #     args = arg_parser.parse_args()
#     #     args.version = False

#     # return args


def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s.%(msecs)03d500: %(levelname).1s %(name)s.py:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Here you put modules where you want more verbose logging

    for module_name in [__name__, "modelgen", "regressors", "wrappers"]:
        logging.getLogger(module_name).setLevel(logging.INFO)


def main() -> int:
    setup_logging()
    args, help_msg = parse_args()
    if args.version:
        print_versions()
    else:
        if args.command == "gen":
            run_gen.main(args)
        else:
            if args.command is None:
                print("No command given\n")
            else:
                print(f"Unknown command {args.command}\n")
            print(help_msg)


    # elif args.active_learning == 0:
    #
    # elif args.active_learning == 1:
    #     modelgen.main_sample_active_learn(args)
    # elif args.active_learning == -1:
    #     modelgen.main_eval_pareto(args)
    # else:
    #     msg = f"Unsupported value --active-learning={args.active_learning}"
    #     raise ValueError(msg)
    # return 0
