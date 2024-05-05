import os
import shutil
from pydoc import locate


def remove_and_mkdir(path: str, remove_if_exists: bool = True) -> None:
    if remove_if_exists:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.mkdir(path)


def custom_import(name: str):
    mod = locate(name)
    return mod
