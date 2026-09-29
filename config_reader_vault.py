import logging
from functools import lru_cache

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.config_reader import ConfigReader

_logger = logging.getLogger("query_classifier")


class ConfigReaderVault:
    @staticmethod
    @lru_cache(maxsize=128)
    def read_config_parameter(key):
        _logger.debug(f"Reading config parameter: {key}")

        key_vault_name = "kv-agnes2-test-westeu-03"
        kv_uri = f"https://{key_vault_name}.vault.azure.net"
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=kv_uri, credential=credential)
        return client.get_secret(key).value
