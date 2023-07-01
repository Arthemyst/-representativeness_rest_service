import os
import pickle

import numpy as np

from tools.tools import (load_models, prepare_data_for_prediction,
                         prepare_data_for_train, remove_old_models)


def test_remove_old_models(tmpdir):
    directory = str(tmpdir.mkdir("test_models"))

    file1 = os.path.join(directory, "model1.pkl")
    file2 = os.path.join(directory, "model2.pkl")
    open(file1, "w").close()
    open(file2, "w").close()

    remove_old_models(directory)

    assert not os.path.exists(file1)
    assert not os.path.exists(file2)


def test_load_models(tmpdir):
    directory = str(tmpdir.mkdir("test_models"))
    model = "MockModel"
    file_path = os.path.join(directory, "model.pkl")

    with open(file_path, "wb") as file:
        pickle.dump(model, file)

    models = load_models(directory)

    assert models == [model]


def test_prepare_data_for_prediction():
    data = [1, 2, 3, 4, 5]
    prepared_data = prepare_data_for_prediction(data)

    assert isinstance(prepared_data, np.ndarray)
    assert prepared_data.shape == (1, 5)
    assert np.array_equal(prepared_data, np.array([[1, 2, 3, 4, 5]]))


def test_prepare_data_for_train():
    data = "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]"

    prepared_data = prepare_data_for_train(data)

    assert isinstance(prepared_data, list)
    assert len(prepared_data) == 3
    assert prepared_data == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
