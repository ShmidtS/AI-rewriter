"""
ModelProvider - Service for model discovery and validation.

Provides a clean interface for:
- Getting available models from the API
- Validating connection to the API
- Getting default model configuration
"""
import logging
from typing import Optional

import requests

from core.services.interfaces import IModelProvider
from core.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class ModelProvider(IModelProvider):
    """
    Service for model discovery and API validation.
    
    Wraps core.api_client functionality with typed Settings integration.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the model provider.
        
        Args:
            settings: Optional Settings instance. Uses global settings if not provided.
        """
        self._settings = settings or get_settings()

    def get_available_models(self) -> list[str]:
        """
        Get list of available models from the API.
        
        Returns:
            List of model names, or default model list if API unavailable.
        """
        base_url = self._settings.get_api_base_url()
        api_key = self._settings.get_api_key()
        default_model = self._settings.get_model_name()
        
        try:
            resp = requests.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            models = [m["id"] for m in data.get("data", []) if "id" in m]
            return sorted(models) if models else [default_model]
        except Exception as e:
            logger.warning(f"Could not fetch model list: {e}")
            return [default_model]

    def validate_connection(self) -> tuple[bool, str]:
        """
        Validate connection to the API.
        
        Returns:
            Tuple of (success, message) where message contains error details if failed.
        """
        base_url = self._settings.get_api_base_url()
        api_key = self._settings.get_api_key()
        
        if not api_key:
            return False, "API key is not configured"
        
        if not base_url:
            return False, "API base URL is not configured"
        
        try:
            # Try to fetch models as a connection test
            resp = requests.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            
            # Parse response to verify it's valid
            data = resp.json()
            if "data" in data:
                model_count = len(data["data"])
                return True, f"Connected successfully. {model_count} models available."
            else:
                return True, "Connected successfully."
                
        except requests.exceptions.ConnectionError as e:
            return False, f"Connection failed: {str(e)}"
        except requests.exceptions.Timeout:
            return False, "Connection timed out"
        except requests.exceptions.HTTPError as e:
            return False, f"HTTP error: {e.response.status_code} - {e.response.text[:200]}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def get_default_model(self) -> str:
        """
        Get the default model name.
        
        Returns:
            Default model name from settings.
        """
        return self._settings.get_model_name()


# Singleton instance for convenience
_model_provider: Optional[ModelProvider] = None


def get_model_provider() -> ModelProvider:
    """
    Get the global ModelProvider instance.
    
    Creates a new instance on first call, then caches it.
    Use this for simple use cases where dependency injection is not needed.
    """
    global _model_provider
    if _model_provider is None:
        _model_provider = ModelProvider()
    return _model_provider
