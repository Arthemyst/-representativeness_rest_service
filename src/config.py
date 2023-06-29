import os

import environ


class CustomEnvironment:
    env = environ.Env(
        MODELS_DIRECTORY=(str, "models"),
    )

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    environ.Env.read_env(os.path.join(BASE_DIR, ".env"))

    _secret_key = env.str("SECRET_KEY")
    _models_directory = env.str("MODELS_DIRECTORY")

    @classmethod
    def get_secret_key(cls) -> str:
        if cls._secret_key is None:
            raise ValueError("Secret key is not provided.")
        return cls._secret_key

    @classmethod
    def get_models_directory(cls) -> str:
        return cls._models_directory
