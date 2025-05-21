# This is parsed by setup.py, so we need to stick to str -> int parsing

# based on evalplus version '0.4.0.dev36', sha ecbe2352bc448bb60e6c3990ac894ff4c4b56ad6

__version__ = "0.1.2"

_ver_major = int(__version__.split(".")[0])
_ver_minor = int(__version__.split(".")[1])
_ver_patch = int(__version__.split(".")[2])

__version_info__ = _ver_major, _ver_minor, _ver_patch
