import toml
from pathlib import Path

class Config:
    def __init__(self, config_path:Path ="config/config.toml"):
        self._config = toml.load(config_path)

    def __getattr__(self, item):
        # Check if item is in the _config dictionary
        if item in self._config:
            value = self._config[item]
            # Return a ConfigWrapper if the value is a dictionary
            if isinstance(value, dict):
                return ConfigWrapper(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

class ConfigWrapper:
    def __init__(self, config_dict):
        self._config_dict = config_dict

    def __getattr__(self, item):
        # Check if item is in the _config_dict
        if item in self._config_dict:
            value = self._config_dict[item]
            # Return a ConfigWrapper if the value is a dictionary
            if isinstance(value, dict):
                return ConfigWrapper(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

config = Config("config/config.toml")
