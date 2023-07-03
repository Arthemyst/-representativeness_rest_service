import pickle
from datetime import datetime, timezone
from typing import List

from pymongo import DESCENDING, MongoClient
from sklearn.linear_model import LinearRegression


def store_model(
    models: List[LinearRegression],
    length_of_object: int,
    training_end_time: str,
    training_start_time: str,
):
    client = MongoClient("mongodb://mongodb:27017")
    db = client["prediction_models"]
    collection = db["models"]
    serialized_models = []
    for model in models:
        serialized_model = pickle.dumps(model)
        serialized_models.append(serialized_model)
    model_data = {
        "models_name": f"models_with_length_of_object_equal_{length_of_object}",
        "model_data_list": serialized_models,
        "training_end_time": training_end_time,
        "training_start_time": training_start_time,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S"),
    }
    collection.insert_one(model_data)

    client.close()


def load_latest_model():
    client = MongoClient("mongodb://mongodb:27017")
    db = client["prediction_models"]
    collection = db["models"]

    model_data = collection.find_one({}, sort=[("training_end_time", DESCENDING)])
    if model_data is None:
        return None
    serialized_models = model_data["model_data_list"]

    models = [pickle.loads(serialized_model) for serialized_model in serialized_models]

    client.close()
    return (
        model_data["models_name"],
        model_data["training_start_time"],
        model_data["training_end_time"],
        models,
    )
