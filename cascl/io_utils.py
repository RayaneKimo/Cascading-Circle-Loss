"""File helpers (JSON / YAML loading and JSON saving).

The custom YAML float resolver matters: PyYAML's SafeLoader parses values
like ``1e-5`` as strings, which would silently break ``learning_rate``.
"""
import json
import logging
import os
import pickle
import re

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)


def load_file(file_path, file_extension=None):
    if file_extension is None:
        _, file_extension = os.path.splitext(file_path)
    if file_extension in [".json", ".yml", ".yaml", ".txt"]:
        mode = "r"
    elif file_extension in [".pkl", ".bin"]:
        mode = "rb"
    else:
        raise ValueError("Un-supported extension")

    with open(file_path, mode) as file:
        logger.info(f"Loading file from {file_path}")
        if file_extension == ".json":
            content = json.load(file)
        elif file_extension in [".yml", ".yaml"]:
            loader = yaml.SafeLoader
            loader.add_implicit_resolver(
                "tag:yaml.org,2002:float",
                re.compile(
                    """^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$""",
                    re.X,
                ),
                list("-+0123456789."),
            )
            content = yaml.load(file, Loader=loader)
        elif file_extension == ".txt":
            content = file.read()
        else:
            content = pickle.load(file)

    return content


def save_file(content, file_path, indent=None):
    _, file_extension = os.path.splitext(file_path)
    if file_extension != ".json":
        raise ValueError("Un-supported extension")
    logger.info(f"Saving file to {file_path}")
    with open(file_path, "w") as file:
        json.dump(content, file, cls=NumpyEncoder, indent=indent, ensure_ascii=False)
