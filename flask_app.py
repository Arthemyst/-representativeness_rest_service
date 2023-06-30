from datetime import datetime
from json import JSONDecodeError
import os
from flask import Flask, jsonify, render_template, request, session
from tasks import train_models_task
from celery.utils.log import get_task_logger
from tools.environment_config import CustomEnvironment
from tools.model_training import calculate_ensemble_prediction
from tasks import train_models_task
from tools.tools import (
    load_models,
    prepare_data_for_prediction,
    prepare_data_for_train,
    remove_old_models,
)

app = Flask(__name__)

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
            session.update(
                {
                    "training_in_progress": True,
                    "training_start_time": datetime.now(),
                    "training_finished": False,
                }
            )

            train_models_task.delay(data)

            session.update(
                {
                    "model_created": True,
                    "training_finished": True,
                    "training_end_time": datetime.now(),
                    "training_in_progress": False,
                    "error_time": False,
                    "error_message": False,
                }
            )

            return render_template(
                "status.html",
                model_created=True,
                training_finished=True,
                training_end_time=session["training_end_time"],
                training_start_time=session["training_start_time"],
            )

        except JSONDecodeError as e:
            session.update(
                {
                    "error_time": datetime.now(),
                    "error_message": str(e),
                    "training_in_progress": False,
                }
            )
            return render_template(
                "train.html",
                bad_data=True,
                msg="Bad data format. Please use list of lists of integers with same length.",
                train_in_progress=session.get("training_in_progress"),
            )

        except ValueError as e:
            session.update(
                {
                    "error_time": datetime.now(),
                    "error_message": str(e),
                    "training_in_progress": False,
                }
            )
            error_msg = "Bad data format. Please use list of lists of integers with same length."
            if str(e) == "Value of k should be less than the number of samples.":
                error_msg = "Please enter more objects to create a model!"
            return render_template(
                "train.html",
                bad_data=True,
                msg=error_msg,
                error_time=session.get("error_time"),
                error_message=session.get("error_message"),
                train_in_progress=session.get("training_in_progress"),
            )

        except Exception as e:
            session.update(
                {
                    "error_time": datetime.now(),
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
                    "error_time": datetime.now(),
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
                    "error_time": datetime.now(),
                    "error_message": str(e),
                }
            )
            return render_template(
                "predict.html",
                bad_data=True,
                model_available=True,
                elements_of_object=elements_of_object,
            )


if __name__ == "__main__":
    app.run(debug=True)
