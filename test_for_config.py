from unittest.mock import patch
import pytest

from tools.environment_config import CustomEnvironment


def test_get_secret_key_no_secret_key():
    with patch.object(CustomEnvironment, "_secret_key", new=None):
        with pytest.raises(ValueError):
            CustomEnvironment.get_secret_key()


@pytest.mark.parametrize("secret_key", ["test_secret_key", "another_secret_key"])
def test_get_secret_key(secret_key):
    with patch.object(CustomEnvironment, "_secret_key", new=secret_key):
        assert CustomEnvironment.get_secret_key() == secret_key


@pytest.mark.parametrize(
    "models_directory", ["test_models_directory_1", "test_models_directory_2"]
)
def test_get_models_directory(models_directory):
    with patch.object(CustomEnvironment, "_models_directory", new=models_directory):
        assert CustomEnvironment.get_models_directory() == models_directory
