from celery_worker import app
from tools.model_training import create_models
from datetime import datetime
import json

@app.task
def train_models_task(data, serialized_session):
    session = json.loads(serialized_session)
    session["model_created"] = False
    try:
        create_models(data)
        session["model_created"] = True
        session["training_finished"] = True
        session["training_end_time"] = str(datetime.now())
        session["training_in_progress"] = False
        session["error_msg_for_train_page"] = None
        session["error_time"] = False
        session["error_message"] = False
        updated_session = json.dumps(session)
        return updated_session

    except ValueError as e:
        error_msg = "Bad data format. Please use list of lists of integers with the same length."
        if str(e) == "Value of k should be less than the number of samples.":
            error_msg = "Please enter more objects to create a model!"
        session["model_created"] = False
        session["training_finished"] = False
        session["training_end_time"] = None
        session["training_in_progress"] = False
        session["error_msg_for_train_page"] = error_msg
        session["error_time"] = str(datetime.now())
        session["error_message"] = str(e)
        updated_session = json.dumps(session)
        return updated_session

