import pytest
from flask_app import app


@pytest.fixture
def client():
    with app.test_client() as client:
        with app.app_context():
            yield client


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Home page" in response.data


def test_train_get(client):
    response = client.get("/train")
    assert response.status_code == 200
    assert b"Train Model" in response.data

def test_predict_get(client):
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Predict" in response.data

def test_status(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert b"Training Status" in response.data
