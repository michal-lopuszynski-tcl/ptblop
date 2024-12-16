from typing import Any


def get_type_name(o: Any) -> str:
    to = type(o)
    return to.__module__ + "." + to.__name__
