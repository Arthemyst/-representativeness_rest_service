from datetime import datetime, timezone
from json import JSONDecodeError

from celery import Celery
from flask import (Flask, jsonify, redirect, render_template, request, session,
                   url_for)

from models_database import load_latest_model, store_model
from tools.environment_config import CustomEnvironment
from tools.model_training import calculate_ensemble_prediction, create_models
from tools.tools import convert_list_to_numpy_array, convert_string_to_list

app = Flask(__name__)
app.config["CELERY_BROKER_URL"] = "redis://redis:6379/0"
app.config["result_backend"] = "redis://redis:6379/0"
app.config["broker_connection_retry_on_startup"] = True

celery = Celery(app.name, broker=app.config["CELERY_BROKER_URL"])
celery.conf.update(app.config)

app.secret_key = CustomEnvironment.get_secret_key()
app.config["SESSION_PERMANENT"] = False
models_directory = CustomEnvironment.get_models_directory()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
def train():
    models_with_names = load_latest_model()
    if models_with_names:
        session["model_created"] = True

    if request.method == "GET":
        task_id = session.get("task_id")
        session["training_in_progress"] = False

        if task_id:
            task = train_models_task.AsyncResult(task_id)
            session["training_in_progress"] = not task.ready()

        return render_template(
            "train.html",
            training_in_progress=session.get("training_in_progress", False),
            training_start_time=session.get("training_start_time", False),
            model_created=session.get("model_created", False),
        )

    if request.method == "POST":
        session["training_started"] = True

        try:
            data = request.form["data"]
            data = convert_string_to_list(data)

            if not isinstance(data, list):
                session.update(
                    {
                        "training_in_progress": False,
                        "error_message": False,
                    }
                )
                return render_template(
                    "train.html",
                    msg=session.get("error_message"),
                    model_created=session.get("model_created", False),
                )

            session.update(
                {
                    "training_in_progress": True,
                    "training_start_time": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S"),
                    "training_finished": False,
                }
            )
            elements_of_object = len(data[0])
            session["elements_of_object"] = elements_of_object
            task = train_models_task.delay(
                data, elements_of_object, session["training_start_time"]
            )
            session["task_id"] = task.id
            return redirect(url_for("train_status", task_id=str(task.id)))

        except JSONDecodeError as e:
            session.update(
                {
                    "error_time": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S"),
                    "error_message": str(e),
                    "training_in_progress": False,
                }
            )
            return render_template(
                "train.html",
                msg="Bad data format. Please use list of lists of integers with the same length.",
                train_in_progress=session.get("training_in_progress"),
                model_created=session.get("model_created", False),
            )

        except ValueError as e:
            session.update(
                {
                    "error_time": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S"),
                    "error_message": str(e),
                    "training_in_progress": False,
                }
            )
            error_msg = "Bad data format. Please use list of lists of integers with the same length."
            if str(e) == "Value of k should be less than the number of samples.":
                error_msg = "Please enter more objects to create a model!"
            return render_template(
                "train.html",
                msg=error_msg,
                train_in_progress=session.get("training_in_progress"),
                model_created=session.get("model_created", False),
            )

        except Exception as e:
            session.update(
                {
                    "error_time": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S"),
                    "error_message": str(e),
                    "training_in_progress": False,
                }
            )
            return render_template(
                "train.html",
                msg="Bad data format. Please use list of lists of integers with the same length.",
                train_in_progress=session.get("training_in_progress", False),
                model_created=session.get("model_created", False),
            )


@app.route("/status", methods=["GET"])
def status():
    models_with_names = load_latest_model()
    if models_with_names:
        session["model_created"] = True
        session["training_start_time"] = models_with_names[1]
        session["training_end_time"] = models_with_names[2]

    training_started = session.get("training_started", False)
    training_in_progress = session.get("training_in_progress", False)
    training_end_time = session.get("training_end_time", False)
    model_created = session.get("model_created", False)
    training_start_time = session.get("training_start_time", False)
    error_message = session.get("error_message", False)
    error_time = session.get("error_time", False)
    task_id = session.get("task_id", False)
    model_created = session.get("model_created", False)
    return render_template(
        "status.html",
        training_started=training_started,
        training_in_progress=training_in_progress,
        training_end_time=training_end_time,
        training_start_time=training_start_time,
        error_message=error_message,
        error_time=error_time,
        task_id=task_id,
        model_created=model_created,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    session["prediction"] = False
    models_with_names = load_latest_model()
    if not models_with_names:
        session["model_created"] = False
        return render_template("predict.html", model_available=False)
    
    models = models_with_names[3]
    models_name = models_with_names[0]
    elements_of_object = models_name.split("_")[-1]
    
    if request.method == "GET":
        return render_template(
            "predict.html",
            model_available=True,
            elements_of_object=elements_of_object,
        )

    if request.method == "POST":
        try:
            data = request.form["data"]
            list_of_objects = convert_string_to_list(data)

            list_of_predictions_for_objects = []
            for object in list_of_objects:
                object_prepared = convert_list_to_numpy_array(object)
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
            session.update(
                {
                    "error_time": None,
                    "error_message": None,
                    "prediction": str(list_of_predictions_for_objects),
                }
            )

            return render_template(
                "predict.html",
                prediction=list_of_predictions_for_objects,
                model_available=True,
                elements_of_object=elements_of_object,
            )
        except ValueError as e:
            session.update(
                {
                    "error_time": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S"),
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
                    "error_time": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S"),
                    "error_message": str(e),
                }
            )
            return render_template(
                "predict.html",
                bad_data=True,
                model_available=True,
                elements_of_object=elements_of_object,
            )


@app.route("/train_status/<task_id>", methods=["GET"])
def train_status(task_id):
    task_result = celery.AsyncResult(task_id)

    if task_result.ready():
        session_results = task_result.get()
        session.update(session_results)

        if session.get("error_msg_for_train_page") == "Please enter more objects to create a model!":
            return render_template(
                "train.html",
                msg=session.get("error_msg_for_train_page", False),
                training_in_progress=session.get("training_in_progress", False),
                model_created=session.get("model_created", False),
                training_finished=session.get("training_finished", False),
                training_start_time=session.get("training_start_time", False),
            )
        else:
            return redirect(url_for("status"))

    return redirect(url_for("status"))



@celery.task
def train_models_task(data, elements_of_object, training_start_time):
    session_results = {
        "model_created": False,
        "training_finished": False,
        "training_end_time": None,
        "training_in_progress": False,
        "error_msg_for_train_page": None,
        "error_time": False,
        "error_message": False,
    }

    try:
        models = create_models(data)
        session_results.update({
            "model_created": True,
            "training_finished": True,
            "training_end_time": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S"),
            "training_in_progress": False,
            "error_msg_for_train_page": None,
            "error_time": False,
            "error_message": False,
        })
        store_model(
            models,
            elements_of_object,
            session_results["training_end_time"],
            training_start_time,
        )
    except ValueError as e:
        error_msg = "Bad data format. Please use a list of lists of integers with the same length."
        if str(e) == "Value of k should be less than the number of samples.":
            error_msg = "Please enter more objects to create a model!"
        session_results.update({
            "training_finished": False,
            "error_msg_for_train_page": error_msg,
            "error_time": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S"),
            "error_message": str(e),
        })
    except Exception as e:
        error_msg = str(e)
        if str(e) == "Value of k should be less than the number of samples.":
            error_msg = "Please enter more objects to create a model!"
        session_results.update({
            "training_finished": False,
            "error_msg_for_train_page": error_msg,
            "error_time": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S"),
            "error_message": str(e),
        })
            

    return session_results


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
