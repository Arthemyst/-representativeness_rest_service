from flask import Flask, jsonify, request, render_template, session
from datetime import datetime
from model_training import create_models, calculate_ensemble_prediction
import os
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "your_secret_key_here"
app.config["SESSION_PERMANENT"] = False


@app.route("/clear_session", methods=["GET"])
def clear_session():
    session.clear()
    return "Session cleared"


@app.route("/train", methods=["GET", "POST"])
def train():
    directory = "models"
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
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
            session["training_start_time"] = datetime.now().isoformat()
            session["training_finished"] = False
            create_models(data)
            session["model_created"] = True
            session["training_finished"] = True
            session["training_end_time"] = datetime.now().isoformat()
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
    directory = "models"
    models = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, "rb") as file:
            model = pickle.load(file)
            models.append(model)
    if not models:
        return render_template("predict.html", model_available=False)

    if request.method == "GET":
        return render_template("predict.html", model_available=True)

    if request.method == "POST":
        try:
            data = request.form["data"]
            session["bad_data"] = False
            # Convert the string to a list of integers
            data = str(data)
            data = list(map(int, data.strip("[]").split(",")))

            session["check"] = str(list(data))
            # Reshape the data to a 2D array
            data = np.array(data)

            # Reshape the data to a 2D array with a single row
            data = data.reshape(1, -1)

            # session["output_data"] = data
            if not isinstance(data, np.ndarray):
                session["bad_data"] = True
                session["type_of_data"] = str(type(data))
                return render_template(
                    "predict.html",
                    prediction=False,
                    bad_data=True,
                    model_available=True,
                )

            ensembled_prediction = calculate_ensemble_prediction(models, data)
            ensembled_prediction_serializable = ensembled_prediction.tolist()[0]
            session["prediction"] = ensembled_prediction_serializable

            return render_template(
                "predict.html",
                prediction=ensembled_prediction_serializable,
                model_available=True,
            )
        except Exception as e:
            return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
