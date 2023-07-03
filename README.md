# representativeness_rest_service
## Introduction
Application for training and using prediction models. Celery with Redis was used to perform the task of training the model asynchronously.

## Images from website

Home page

![image](https://github.com/Arthemyst/representativeness_rest_service/assets/59807704/18a9f0ef-4e68-483a-996a-6b5952de92cc)

Model training page

![image](https://github.com/Arthemyst/representativeness_rest_service/assets/59807704/ce00ae8d-d096-480d-9de3-b1180d8cb90d)

Status page

![image](https://github.com/Arthemyst/representativeness_rest_service/assets/59807704/8aca5c10-2df9-4560-8142-7014d269e89d)


Prediction page

![image](https://github.com/Arthemyst/representativeness_rest_service/assets/59807704/0cec1044-26d0-4c6e-99fd-873150b676d8)


## Setup
The first thing to do is to clone the repository:
```sh
$ git clone https://github.com/Arthemyst/representativeness_rest_service.git
$ cd representativeness_rest_service
```
This project requires Python 3.8 or later.

Please create file .env in root directory. The file format can be understood from the example below::

```sh
SECRET_KEY=your_secret_key
```

### Start docker:

Application runs on docker. To run app you need to install docker on your machine. Please run docker-compose to install dependiences and run application:

```sh
$ docker-compose up --build
```

Docker:
- flask application
- celery
- redis
- pytest
