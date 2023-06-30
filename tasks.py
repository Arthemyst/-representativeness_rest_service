from celery_worker import app
from tools.model_training import create_models


@app.task
def train_models_task(data):
    create_models(data)
