import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from model_training import (
    calculate_distance,
    calculate_representativeness,
    calculate_ensemble_prediction,
    train_model,
    representative_learning,
    convert_data_to_numpy_array,
    find_optimal_L,
    random_split_data,
    create_models,
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
        assert False  # This line should not be reached
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
        assert False  # This line should not be reached
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


def test_create_models(tmpdir):
    data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    k = 2
    expected_model_paths = ["model_0", "model_1"]

    create_models(data, k)

    directory = "models"
    model_paths = os.listdir(directory)
    assert len(model_paths) == len(expected_model_paths)
    for expected_path in expected_model_paths:
        assert expected_path in model_paths

    for model_path in model_paths:
        path = os.path.join(directory, model_path)
        os.remove(path)
    os.rmdir(directory)
