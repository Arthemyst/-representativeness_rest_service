import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import cross_val_score
import pickle
import os


def calculate_distance(X, k):
    neighbor = NearestNeighbors(n_neighbors=k)
    neighbor.fit(X)
    distances, _ = neighbor.kneighbors(X)
    return np.mean(distances[:, -1])


def calculate_representativeness(X, k):
    distance = calculate_distance(X, k)
    return 1 / (1 + distance)


def calculate_ensemble_prediction(models, X):
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0)


def train_model(subset, k):
    representativeness = [
        calculate_representativeness(subset, k) for _ in range(len(subset))
    ]
    model = LinearRegression()
    model.fit(subset, representativeness)
    return model


def representative_learning(subsets, k):
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


def convert_data_to_numpy_array(data):
    data = np.array(data)
    if data.size == 0:
        raise ValueError("Input data must have at least one sample.")

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def find_optimal_L(X, k, min_L=1, max_L=5, cv=5):
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


def random_split_data(X, L):
    subsets = np.array_split(X, L)
    return subsets


def create_models(data, k=3):
    X = convert_data_to_numpy_array(data)
    if k >= X.shape[0]:
        raise ValueError("Value of k should be less than the number of samples.")
    L = 2
    subsets = random_split_data(X, L)
    models = representative_learning(subsets, k)
    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, model in enumerate(models):
        model_path = os.path.join(directory, f"{model}_{i}")
        with open(model_path, "wb") as file:
            pickle.dump(model, file)


data = [
    [1, 3, 4, 5, 6, 7, 7, 5, 6, 7, 7],
    [5, 2, 6, 2, 1, 8, 8, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 35, 6, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 34, 43, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 345, 35, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 345, 35, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 324, 234, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 34, 43, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 34, 43, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 34, 43, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 324, 234, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 34, 43, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 34, 43, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 34, 43, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 324, 234, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 34, 43, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 34, 43, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 34, 43, 5, 6, 7, 7],
    [1, 3, 4, 5, 6, 7, 7, 5, 6, 7, 7],
    [5, 2, 6, 2, 1, 8, 8, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 35, 6, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 34, 43, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 345, 35, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 345, 35, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 324, 234, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 324, 234, 5, 6, 7, 7],
    [9, 8, 3, 6, 2, 324, 234, 5, 6, 7, 7],
]
# models = create_models(data)
# ensemble_prediction = calculate_ensemble_prediction(
#     models, [[9, 1, 3, 2, 2, 2635363636632, 0, 5, 6, 7, 7]]
# )
# print(ensemble_prediction[0])
