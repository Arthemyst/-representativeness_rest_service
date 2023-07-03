import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import numpy as np
from models_database import store_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

from tools.environment_config import CustomEnvironment


def calculate_distance(X: np.ndarray, k: int) -> Union[float, np.float64]:
    neighbor = NearestNeighbors(n_neighbors=k)
    neighbor.fit(X)
    distances, _ = neighbor.kneighbors(X)
    return np.mean(distances[:, -1])


def calculate_representativeness(X: np.ndarray, k: int) -> Union[float, np.float64]:
    distance = calculate_distance(X, k)
    return 1 / (1 + distance)


def calculate_ensemble_prediction(models: List[LinearRegression], X: np.ndarray):
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0)


def train_model(subset: np.ndarray, k: int) -> LinearRegression:
    representativeness = [
        calculate_representativeness(subset, k) for _ in range(len(subset))
    ]
    model = LinearRegression()
    model.fit(subset, representativeness)
    return model


def representative_learning(
    subsets: List[np.ndarray], k: int
) -> List[LinearRegression]:
    models = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for subset in subsets:
            if k >= subset.shape[0]:
                raise ValueError(
                    "Value of k should be less than the number of samples."
                )
            future = executor.submit(train_model, subset, k)
            futures.append(future)

        for future in futures:
            model = future.result()
            models.append(model)

    return models


def convert_data_to_numpy_array(data: list) -> np.ndarray:
    data = np.array(data)
    if data.size == 0:
        raise ValueError("Input data must have at least one sample.")

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def find_optimal_L(
    X: np.ndarray, k: int, min_L: int = 1, max_L: int = 5, cv: int = 5
) -> int:
    results = []
    for L in range(min_L, max_L + 1):
        subsets = random_split_data(X, L)
        models = representative_learning(subsets, k)
        ensemble_prediction = calculate_ensemble_prediction(models, X)
        scores = cross_val_score(
            LinearRegression(),
            ensemble_prediction.reshape(-1, 1),
            X[:, 0].reshape(-1, 1),
            cv=cv,
        )
        average_score = np.mean(scores)
        results.append((L, average_score))

    optimal_L = max(results, key=lambda x: x[1])[0]
    return optimal_L


def random_split_data(X: np.ndarray, L: int) -> List[np.ndarray]:
    subsets = np.array_split(X, L)
    return subsets


def create_models(data: list, k: int = 5) -> List[LinearRegression]:
    X = convert_data_to_numpy_array(data)
    if k >= X.shape[0]:
        raise ValueError("Value of k should be less than the number of samples.")
    L = 2
    subsets = random_split_data(X, L)
    models = representative_learning(subsets, k)
    return models
