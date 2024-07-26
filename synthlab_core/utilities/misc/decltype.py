import numpy as np
from PIL import Image
from io import BytesIO
from pydoc import locate


def type2readble(inp):
    if isinstance(inp, type):
        return inp.__name__

    if isinstance(inp, (list, tuple)):
        return [type2readble(i) for i in inp]

    raise ValueError(f"Why do we have this fucking stupid input type? {inp}")


def is_image(buffer: bytes):
    try:
        if isinstance(buffer, str):
            buffer = buffer.encode()

        buffer = BytesIO(buffer)
        Image.open(buffer)
    except Exception as err:
        print("ERROR", err)
        return False

    return True


def is_nparray(buffer: bytes):
    try:
        if isinstance(buffer, str):
            buffer = buffer.encode()

        buffer = BytesIO(buffer)
        np.load(buffer)
    except Exception as err:
        print("ERROR", err)
        return False

    return True


__custom_type_declaration = ["synthlab_core.common.atomic."]


def str2type(s: str):
    global __custom_type_declaration

    if isinstance(s, (list, tuple)):
        return [str2type(x) for x in s]

    x = locate(s)

    if x is not None:
        return x

    for item in __custom_type_declaration:
        x = locate(item + s)

        if x is not None:
            return x

    return None


def type_compatible(src, dest):
    if isinstance(src, (str, list, tuple)):
        src = str2type(src)

    if isinstance(dest, (str, list, tuple)):
        dest = str2type(dest)

    if isinstance(dest, (list, tuple)):
        return any(issubclass(src, x) for x in dest)

    return issubclass(src, dest)
