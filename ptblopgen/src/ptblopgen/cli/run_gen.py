import pathlib
from typing import Any

from ptblop import __version__ as ptblop_version

from .._version import __version__ as ptblopgen_version


def main(config_raw: dict[str, Any], output_path: pathlib.Path):

    print("Generating!!!")
