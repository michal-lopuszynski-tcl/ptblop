# This is parsed by setup.py, so we need to stick to str -> int parsing

# Initially based on versio 0.8.48 of ptdeco trainer

__version__ = "0.1.0"

_ver_major = int(__version__.split(".")[0])
_ver_minor = int(__version__.split(".")[1])
_ver_patch = int(__version__.split(".")[2])

__version_info__ = _ver_major, _ver_minor, _ver_patch
