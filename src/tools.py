import os
import pickle

import numpy as np


def remove_old_models(directory: str) -> None:
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)


def load_models(directory: str) -> list:
    models = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, "rb") as file:
            model = pickle.load(file)
            models.append(model)
    return models


def prepare_data(data: str) -> np.ndarray:
    data = list(map(int, data.strip("[]").split(",")))
    data = np.array(data)
    data = data.reshape(1, -1)
    return data
