FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN python -m venv venv
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["venv/bin/python", "flask_app.py"]
