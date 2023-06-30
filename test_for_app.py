import pytest
from flask import session
from flask_app import app
from tools.environment_config import CustomEnvironment
from unittest.mock import patch


@pytest.fixture
def client():
    with app.test_client() as client:
        with app.app_context():
            yield client


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Home page" in response.data


def test_clear_session(client):
    with client.session_transaction() as sess:
        sess["key"] = "value"

    response = client.get("/clear_session")
    assert response.status_code == 200
    assert session.get("key") is None


@pytest.mark.parametrize("models_directory", ["test_models_directory"])
def test_train_get(client, models_directory):
    with patch.object(CustomEnvironment, "_models_directory", new=models_directory):
        response = client.get("/train")
        assert response.status_code == 200
        assert b"Train Model" in response.data


@pytest.mark.parametrize("models_directory", ["test_models_directory"])
def test_train_post_bad_data(client, models_directory):
    with patch.object(CustomEnvironment, "_models_directory", new=models_directory):
        data = {"data": "invalid_data"}

        response = client.post("/train", data=data)
        assert response.status_code == 200
        assert (
            b"Bad data format. Please use list of lists of integers with same length."
            in response.data
        )


@pytest.mark.parametrize("models_directory", ["test_models_directory"])
def test_train_post_valid_data(client, models_directory):
    with patch.object(CustomEnvironment, "_models_directory", new=models_directory):
        data = {"data": "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]"}

        response = client.post("/train", data=data)
        assert response.status_code == 200
        assert b"Please enter more objects to create a model!" in response.data


def test_status(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert b"Training Status" in response.data


@pytest.mark.parametrize("models_directory", ["test_models_directory"])
def test_predict_get_with_trained_model(client, models_directory):
    with patch.object(CustomEnvironment, "_models_directory", new=models_directory):
        response = client.get("/clear_session")
        data = {
            "data": "[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9],[1, 2, 3], [4, 5, 6], [7, 8, 9],[1, 2, 3], [4, 5, 6], [7, 8, 9],[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]"
        }
        response = client.post("/train", data=data)
        response = client.get("/predict")
        assert response.status_code == 200
        assert b"3" in response.data


def test_predict_post_bad_data(client):
    response = client.get("/clear_session")
    data = {"data": "invalid_data"}

    response = client.post("/predict", data=data)
    assert response.status_code == 200
    assert b"Bad data" in response.data


@pytest.mark.parametrize("models_directory", ["test_models_directory"])
def test_predict_post_valid_data(client, models_directory):
    with patch.object(CustomEnvironment, "_models_directory", new=models_directory):
        response = client.get("/clear_session")
        data = {
            "data": "[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9],[1, 2, 3], [4, 5, 6], [7, 8, 9],[1, 2, 3], [4, 5, 6], [7, 8, 9],[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]"
        }
        response = client.post("/train", data=data)
        data = {"data": "[[1, 2, 3]]"}

        response = client.post("/predict", data=data)
        assert response.status_code == 200
        assert b"0.161" in response.data
