from datetime import datetime

import numpy as np
from flask import Flask, jsonify, render_template, request, session

from config import CustomEnvironment
from model_training import calculate_ensemble_prediction, create_models
from tools import load_models, prepare_data, remove_old_models

app = Flask(__name__)
app.secret_key = CustomEnvironment.get_secret_key()
app.config["SESSION_PERMANENT"] = False
models_directory = CustomEnvironment.get_models_directory()


@app.route("/", methods=["GET"])
def index():
    return "Hello, world!"


@app.route("/clear_session", methods=["GET"])
def clear_session():
    session.clear()
    return "Session cleared"


@app.route("/train", methods=["GET", "POST"])
def train():
    remove_old_models(models_directory)

    if request.method == "GET":
        return render_template(
            "train.html",
            training_in_progress=session.get("training_in_progress", False),
            training_start_time=session.get("training_start_time"),
        )

    if session.get("training_in_progress"):
        return jsonify(
            {
                "status": "Training still in progress",
                "start_time": session["training_start_time"],
            }
        )

    if request.method == "POST":
        try:
            data = request.get_json()
            if not isinstance(data, list):
                return jsonify({"error": "Invalid data format"})

            session["training_in_progress"] = True
            session["training_start_time"] = datetime.now()
            session["training_finished"] = False

            create_models(data)

            session["model_created"] = True
            session["training_finished"] = True
            session["training_end_time"] = datetime.now()
            session["training_in_progress"] = False

            return jsonify(
                {
                    "status": "Training started",
                    "start_time": session["training_start_time"],
                }
            )
        except Exception as e:
            return jsonify(
                {
                    "error": "Error during training",
                    "start_time": session["training_start_time"],
                    "erro_time": datetime.now(),
                    "message": str(e),
                }
            )


@app.route("/session_items", methods=["GET"])
def session_items():
    items = dict(session)
    return jsonify(items)


@app.route("/status", methods=["GET"])
def status():
    if session.get("training_in_progress", False):
        return render_template(
            "status.html",
            training_in_progress=True,
            training_start_time=session["training_start_time"],
        )
    elif session.get("training_end_time"):
        return render_template(
            "status.html",
            training_in_progress=False,
            model_created=session["model_created"],
            training_end_time=session["training_end_time"],
            training_start_time=session["training_start_time"],
        )
    else:
        return render_template("status.html", training_in_progress=False)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    session["prediction"] = False
    models = load_models(models_directory)
    if not models:
        return render_template("predict.html", model_available=False)

    if request.method == "GET":
        return render_template("predict.html", model_available=True)

    if request.method == "POST":
        try:
            data = request.form["data"]
            data = prepare_data(data)

            if not isinstance(data, np.ndarray):
                return render_template(
                    "predict.html",
                    prediction=False,
                    bad_data=True,
                    model_available=True,
                )

            ensembled_prediction = calculate_ensemble_prediction(models, data)
            ensembled_prediction_serializable = round(ensembled_prediction[0], 3)
            session["prediction"] = ensembled_prediction_serializable

            return render_template(
                "predict.html",
                prediction=ensembled_prediction_serializable,
                model_available=True,
            )
        except ValueError:
            return render_template(
                "predict.html",
                bad_data=True,
                model_available=True,
            )
        except Exception as e:
            return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
