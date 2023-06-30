from celery_worker import app
from tools.model_training import create_models
from datetime import datetime


@app.task
def train_models_task(data):
    try:
        create_models(data)
        return {
            "model_created": True,
            "training_finished": True,
            "training_end_time": datetime.now(),
            "training_in_progress": False,
            "error_msg_for_train_page": None,
            "error_time": False,
            "error_message": False,
        }

    except ValueError as e:
        error_msg = (
            "Bad data format. Please use list of lists of integers with same length."
        )
        if str(e) == "Value of k should be less than the number of samples.":
            error_msg = "Please enter more objects to create a model!"
        return {
            "model_created": False,
            "training_finished": False,
            "training_end_time": None,
            "training_in_progress": False,
            "error_msg_for_train_page": error_msg,
            "error_time": datetime.now(),
            "error_message": str(e),
        }
