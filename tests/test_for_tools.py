import os
import pickle

import numpy as np
from tools.tools import convert_list_to_numpy_array, convert_string_to_list


def test_convert_list_to_numpy_array():
    data = [1, 2, 3, 4, 5]
    prepared_data = convert_list_to_numpy_array(data)

    assert isinstance(prepared_data, np.ndarray)
    assert prepared_data.shape == (1, 5)
    assert np.array_equal(prepared_data, np.array([[1, 2, 3, 4, 5]]))


def test_convert_string_to_list():
    data = "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]"

    prepared_data = convert_string_to_list(data)

    assert isinstance(prepared_data, list)
    assert len(prepared_data) == 3
    assert prepared_data == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
