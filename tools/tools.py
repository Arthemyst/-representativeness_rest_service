import json
import os
import pickle
from tools.model_training import calculate_ensemble_prediction
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


def prepare_data_for_prediction(data: list) -> np.ndarray:
    data = np.array(data)
    data = data.reshape(1, -1)
    return data


def prepare_data_for_train(data: str) -> list:
    data_list = json.loads(data)
    return list(data_list)


# data = "[[1,2,3],[4, 5, 6]]"

# list_of_objects = prepare_data_for_train(data)
# print(type(list_of_objects))
# models = load_models("models")
# list_of_predictions_for_objects = []
# for object in list_of_objects:
#     object_prepared = prepare_data_for_prediction(object)
#     if not isinstance(object, list):
#         print("error")
#         break

#     ensembled_prediction = calculate_ensemble_prediction(models, object_prepared)
#     ensembled_prediction_serializable = round(ensembled_prediction[0], 3)
#     list_of_predictions_for_objects.append(ensembled_prediction_serializable)
# print(list_of_predictions_for_objects)
