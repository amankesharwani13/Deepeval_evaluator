import os


class ConfigReader:
    @staticmethod
    def read_config_parameter(key, default=None):
        return os.getenv(key, default)
