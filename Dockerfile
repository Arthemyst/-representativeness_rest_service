FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytest

COPY . .

CMD ["python", "flask_app.py"]
