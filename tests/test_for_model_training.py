import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.linear_model import LinearRegression
from tools.model_training import (
    calculate_distance,
    calculate_ensemble_prediction,
    calculate_representativeness,
    convert_data_to_numpy_array,
    find_optimal_L,
    random_split_data,
    representative_learning,
    train_model,
)


def test_calculate_distance():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    k = 2
    expected_distance = 2.8284271247461903
    assert calculate_distance(X, k) == expected_distance


def test_calculate_representativeness():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    k = 2
    expected_representativeness = 0.2612038749637414
    assert calculate_representativeness(X, k) == expected_representativeness


def test_calculate_ensemble_prediction():
    models = [
        LinearRegression().fit(np.array([[1, 2], [3, 4]]), np.array([1, 2])),
        LinearRegression().fit(np.array([[5, 6], [7, 8]]), np.array([3, 4])),
    ]
    X = np.array([[1, 2], [3, 4]])
    expected_prediction = np.array([1.0, 2.0])
    np.testing.assert_allclose(
        calculate_ensemble_prediction(models, X), expected_prediction, atol=1e-8
    )


def test_train_model():
    subset = np.array([[1, 2], [3, 4], [5, 6]])
    k = 2

    model = train_model(subset, k)

    assert isinstance(model, LinearRegression)
    assert model.coef_ is not None
    assert model.intercept_ is not None


def test_representative_learning():
    subsets = [
        np.array([[1, 2], [3, 4], [9, 10]]),
        np.array([[5, 6], [7, 8], [9, 10]]),
    ]
    k = 2

    models = representative_learning(subsets, k)

    assert len(models) == len(subsets)
    for model in models:
        assert isinstance(model, LinearRegression)
        assert model.coef_ is not None
        assert model.intercept_ is not None


def test_convert_data_to_numpy_array():
    data = [1, 2, 3, 4, 5]

    result = convert_data_to_numpy_array(data)

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array(data).reshape(-1, 1))


def test_convert_data_to_numpy_array_empty_data():
    data = []

    try:
        convert_data_to_numpy_array(data)
    except ValueError as e:
        assert str(e) == "Input data must have at least one sample."


def test_convert_data_to_numpy_array_2d_data():
    data = [[1, 2], [3, 4], [5, 6]]

    result = convert_data_to_numpy_array(data)

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array(data))


def test_find_optimal_L():
    X = np.array(
        [
            [1, 2, 3],
            [3, 4, 5],
            [5, 6, 7],
            [7, 8, 9],
            [9, 10, 11],
            [11, 12, 12],
            [13, 14, 15],
            [1, 2, 3],
            [3, 4, 5],
            [5, 6, 7],
            [7, 8, 9],
            [9, 10, 11],
            [11, 12, 13],
            [13, 14, 15],
            [11, 12, 13],
            [13, 14, 15],
        ]
    )
    k = 2
    min_L = 1
    max_L = 5
    cv = 3

    optimal_L = find_optimal_L(X, k, min_L, max_L, cv)

    assert optimal_L >= min_L
    assert optimal_L <= max_L


def test_find_optimal_L_empty_X():
    X = np.empty((0, 2))
    k = 2
    min_L = 1
    max_L = 5
    cv = 3

    try:
        find_optimal_L(X, k, min_L, max_L, cv)
    except ValueError as e:
        assert str(e) == "Value of k should be less than the number of samples."


def test_find_optimal_L_k_greater_than_samples():
    X = np.array([[1, 2], [3, 4]])
    k = 3
    min_L = 1
    max_L = 5
    cv = 3

    try:
        find_optimal_L(X, k, min_L, max_L, cv)
    except ValueError as e:
        assert str(e) == "Value of k should be less than the number of samples."


def test_random_split_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    L = 3

    subsets = random_split_data(X, L)

    assert len(subsets) == L
    assert all(isinstance(subset, np.ndarray) for subset in subsets)
    assert all(subset.shape[0] == X.shape[0] // L for subset in subsets)


def test_random_split_data_empty_X():
    X = np.empty((0, 2))
    L = 3

    subsets = random_split_data(X, L)

    assert len(subsets) == L
    assert all(isinstance(subset, np.ndarray) for subset in subsets)
    assert all(subset.shape[0] == 0 for subset in subsets)


def test_random_split_data_L_greater_than_samples():
    X = np.array([[1, 2], [3, 4]])
    L = 3

    try:
        random_split_data(X, L)

    except ValueError as e:
        assert str(e) == "Value of k should be less than the number of samples."
