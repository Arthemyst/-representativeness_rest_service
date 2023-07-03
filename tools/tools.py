import json
import os
import pickle

import numpy as np


def convert_list_to_numpy_array(data: list) -> np.ndarray:
    data = np.array(data)
    data = data.reshape(1, -1)
    return data


def convert_string_to_list(data: str) -> list:
    data_list = json.loads(data)
    return list(data_list)
