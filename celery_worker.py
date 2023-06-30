from celery import Celery

app = Celery(
    "celery_worker",
    broker="amqp://guest:guest@rabbitmq:5672//",
    backend="rpc://",
    include=["tasks"],
)

if __name__ == "__main__":
    app.start()
