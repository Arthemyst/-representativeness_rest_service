# representativeness_rest_service
## Introduction
Application for training and using prediction models.

## Setup
The first thing to do is to clone the repository:
```sh
$ git clone https://github.com/Arthemyst/representativeness_rest_service.git
$ cd representativeness_rest_service
```
This project requires Python 3.8 or later.

Please create file .env in docker directory. The file format can be understood from the example below::

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
