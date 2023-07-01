from datetime import datetime
from json import JSONDecodeError
import os
import json
from tools.model_training import create_models
from celery import Celery
from flask import Flask, jsonify, render_template, request, session
from tools.environment_config import CustomEnvironment
from tools.model_training import calculate_ensemble_prediction
from tools.tools import (
    load_models,
    prepare_data_for_prediction,
    prepare_data_for_train,
    remove_old_models,
)

app = Flask(__name__)
app.config["CELERY_BROKER_URL"] = "redis://redis:6379/0"
app.config["CELERY_RESULT_BACKEND"] = "redis://redis:6379/0"

celery = Celery(app.name, broker=app.config["CELERY_BROKER_URL"])
celery.conf.update(app.config)

app.secret_key = CustomEnvironment.get_secret_key()
app.config["SESSION_PERMANENT"] = False
models_directory = CustomEnvironment.get_models_directory()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/session_items", methods=["GET"])
def session_items():
    items = dict(session)
    return jsonify(items)


@app.route("/clear_session", methods=["GET"])
def clear_session():
    session.clear()
    return "Session cleared"


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "GET":
        task_id = session.get("task_id")
        session["training_in_progress"] = False

        # If task ID exists, check the status of the task
        if task_id:
            task = train_models_task.AsyncResult(task_id)
            if task.ready():
                # Task is completed, retrieve the result
                # session['result'] = task.get()
                session["training_in_progress"] = False
            else:
                # Task is still in progress
                session["training_in_progress"] = True
        return render_template(
            "train.html",
            training_in_progress=session.get("training_in_progress", False),
            training_start_time=session.get("training_start_time", False),
        )

    if request.method == "POST":
        remove_old_models(models_directory)
        session["model_created"] = False

        try:
            data = request.form["data"]
            data = prepare_data_for_train(data)

            if not isinstance(data, list):
                session.update(
                    {
                        "training_in_progress": False,
                        "error_message": False,
                    }
                )
                return render_template(
                    "train.html", bad_data=True, msg=session.get("error_message")
                )

            session_data = {}
            for key, value in session.items():
                session_data[key] = value
            session.update(
                {
                    "training_in_progress": True,
                    "training_start_time": str(datetime.now()),
                    "training_finished": False,
                }
            )
            session["elements_in_list"] = len(data[0])
            task = train_models_task.delay(data)
            session["task_id"] = task.id
            
   


            return render_template("status.html", task_id=str(task.id))


        except JSONDecodeError as e:
            session.update(
                {
                    "error_time": str(datetime.now()),
                    "error_message": str(e),
                    "training_in_progress": False,
                }
            )
            return render_template(
                "train.html",
                msg="Bad data format. Please use list of lists of integers with same length.",
                train_in_progress=session.get("training_in_progress"),
            )

        except ValueError as e:
            session.update(
                {
                    "error_time": str(datetime.now()),
                    "error_message": str(e),
                    "training_in_progress": False,
                }
            )
            error_msg = "Bad data format. Please use list of lists of integers with same length."
            if str(e) == "Value of k should be less than the number of samples.":
                error_msg = "Please enter more objects to create a model!"
            return render_template(
                "train.html",
                msg=error_msg,
                error_time=session.get("error_time"),
                error_message=session.get("error_message"),
                train_in_progress=session.get("training_in_progress"),
            )

        except Exception as e:
            session.update(
                {
                    "error_time": str(datetime.now()),
                    "error_message": str(e),
                    "training_in_progress": False,
                }
            )
            return render_template(
                "train.html",
                bad_data=True,
                msg="Bad data format. Please use list of lists of integers with same length.",
                error_time=session.get("error_time"),
                error_message=session.get("error_message"),
                train_in_progress=session.get("training_in_progress"),
            )


@app.route("/status", methods=["GET"])
def status():
    training_in_progress = session.get("training_in_progress", False)
    training_end_time = session.get("training_end_time", False)
    model_created = session.get("model_created", False)
    training_start_time = session.get("training_start_time", False)
    error_message = session.get("error_message", False)
    error_time = session.get("error_time", False)

    return render_template(
        "status.html",
        training_in_progress=training_in_progress,
        training_end_time=training_end_time,
        model_created=model_created,
        training_start_time=training_start_time,
        error_message=error_message,
        error_time=error_time,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    session["prediction"] = False
    elements_of_object = session.get("elements_in_list", False)
    if not os.path.exists(models_directory):
        session["model_created"] = False
        training_in_progress = session.get("training_in_progress", False)
        return render_template(
            "status.html",
            model_created=session["model_created"],
            training_in_progress=training_in_progress,
        )

    models = load_models(models_directory)
    if not models:
        return render_template("predict.html", model_available=False)

    if request.method == "GET":
        return render_template(
            "predict.html", model_available=True, elements_of_object=elements_of_object
        )

    if request.method == "POST":
        try:
            data = request.form["data"]
            list_of_objects = prepare_data_for_train(data)

            list_of_predictions_for_objects = []
            for object in list_of_objects:
                object_prepared = prepare_data_for_prediction(object)
                if not isinstance(object, list):
                    return render_template(
                        "predict.html",
                        prediction=False,
                        bad_data=True,
                        model_available=True,
                        elements_of_object=elements_of_object,
                    )

                ensembled_prediction = calculate_ensemble_prediction(
                    models, object_prepared
                )
                ensembled_prediction_serializable = round(ensembled_prediction[0], 3)
                list_of_predictions_for_objects.append(
                    ensembled_prediction_serializable
                )
            session["prediction"] = str(list_of_predictions_for_objects)

            return render_template(
                "predict.html",
                prediction=list_of_predictions_for_objects,
                model_available=True,
                elements_of_object=elements_of_object,
            )
        except ValueError as e:
            session.update(
                {
                    "error_time": str(datetime.now()),
                    "error_message": str(e),
                }
            )
            return render_template(
                "predict.html",
                bad_data=True,
                model_available=True,
                elements_of_object=elements_of_object,
            )
        except Exception as e:
            session.update(
                {
                    "error_time": str(datetime.now()),
                    "error_message": str(e),
                }
            )
            return render_template(
                "predict.html",
                bad_data=True,
                model_available=True,
                elements_of_object=elements_of_object,
            )


@app.route("/status/<task_id>", methods=["GET"])
def train_status(task_id):
    # Retrieve the task result from Celery using the task ID
    task_result = celery.AsyncResult(task_id)

    if task_result.ready():
        # If the task is complete, retrieve the result
        session["training_in_progress"] = True
        session["model_created"] = True
        session["training_finished"] = True

        # Render the status.html template with the task ID, status, and result
        return render_template(
            "status.html",
            task_id=task_id,
            training_in_progress=session["training_in_progress"],
            model_created=session["model_created"],
            training_finished=session["training_finished"],
            training_start_time=session["training_start_time"]
        )
    else:
        session["training_in_progress"] = False
        session["model_created"] = False
        session["training_finished"] = False
        return render_template(
            "status.html",
            task_id=task_id,
            training_in_progress=session["training_in_progress"],
            model_created=session["model_created"],
            training_finished=session["training_finished"],
            training_start_time=session["training_start_time"]
        )

@celery.task
def train_models_task(data):
    session_results = {}

    try:
        create_models(data)
        session_results["model_created"] = True
        session_results["training_finished"] = True
        session_results["training_end_time"] = str(datetime.now())
        session_results["training_in_progress"] = False
        session_results["error_msg_for_train_page"] = None
        session_results["error_time"] = False
        session_results["error_message"] = False
        return session_results

    except ValueError as e:
        error_msg = "Bad data format. Please use list of lists of integers with the same length."
        if str(e) == "Value of k should be less than the number of samples.":
            error_msg = "Please enter more objects to create a model!"
        session_results["model_created"] = False
        session_results["training_finished"] = False
        session_results["training_end_time"] = None
        session_results["training_in_progress"] = False
        session_results["error_msg_for_train_page"] = error_msg
        session_results["error_time"] = str(datetime.now())
        session_results["error_message"] = str(e)
        return session_results

    except Exception as e:
        error_msg = "WRONG"
        if str(e) == "Value of k should be less than the number of samples.":
            error_msg = "Please enter more objects to create a model!"
        session_results["model_created"] = False
        session_results["training_finished"] = False
        session_results["training_end_time"] = None
        session_results["training_in_progress"] = False
        session_results["error_msg_for_train_page"] = error_msg
        session_results["error_time"] = str(datetime.now())
        session_results["error_message"] = str(e)
        return session_results


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
