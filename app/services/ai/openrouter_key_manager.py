"""
app/services/ai/openrouter_key_manager.py
═══════════════════════════════════════════
OpenRouter API Key Rotation Manager with JSON state file.

Features:
- Loads API keys from .env file (OPENROUTER_API_KEY)
- Stores only active key index in JSON file
- Reads the JSON file to get the current active key index
- On key exhaustion (429 error), increments index and updates JSON
- Efficient: no iteration through all keys, just reads from JSON state
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Optional, List

logger = logging.getLogger(__name__)


class OpenRouterKeyManager:
    """
    Manages OpenRouter API key rotation using a JSON state file.
    
    Keys are loaded from .env file (OPENROUTER_API_KEY).
    The JSON file stores only:
    - active_key_index: Index of the current active key
    
    When a key is exhausted (429 error), the index is incremented
    and the next key becomes active.
    """
    
    def __init__(
        self, 
        keys_from_env: List[str],
        state_file_path: str = "app/services/ai/openrouter_key_state.json"
    ):
        """
        Initialize the OpenRouter key manager.
        
        Args:
            keys_from_env: List of API keys loaded from .env
            state_file_path: Path to the JSON state file
        """
        self.state_file_path = Path(state_file_path)
        self._lock = Lock()
        self._keys = keys_from_env
        
        if not self._keys:
            logger.error("No OpenRouter API keys provided from .env")
            raise ValueError("No OpenRouter API keys provided from .env")
        
        # Ensure the state file exists
        if not self.state_file_path.exists():
            logger.warning(f"State file not found, creating: {self.state_file_path}")
            self._create_initial_state()
        
        logger.info(
            f"OpenRouterKeyManager initialized with {len(self._keys)} keys from .env, "
            f"state file: {self.state_file_path}"
        )
    
    def _create_initial_state(self) -> None:
        """Create initial state file with index 0."""
        initial_state = {"active_key_index": 0}
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file_path, 'w') as f:
            json.dump(initial_state, f, indent=2)
        logger.info("Created initial state file")
    
    def _load_state(self) -> dict:
        """
        Load the current state from the JSON file.
        
        Returns:
            Dictionary with 'active_key_index'
        """
        try:
            with open(self.state_file_path, 'r') as f:
                state = json.load(f)
            
            # Validate state structure
            if 'active_key_index' not in state:
                logger.error("Invalid state file structure, missing 'active_key_index'")
                raise ValueError("Invalid state file structure")
            
            return state
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON state file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load state file: {e}")
            raise
    
    def _save_state(self, state: dict) -> None:
        """
        Save the current state to the JSON file.
        
        Args:
            state: Dictionary with 'active_key_index'
        """
        try:
            with open(self.state_file_path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug("State file saved successfully")
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")
            raise
    
    def get_active_key(self) -> str:
        """
        Get the current active API key based on the index in JSON state file.
        
        Returns:
            The active API key string
        """
        with self._lock:
            state = self._load_state()
            active_index = state['active_key_index']
            
            # Validate index
            if active_index < 0 or active_index >= len(self._keys):
                logger.warning(
                    f"Invalid active_key_index {active_index} (total keys: {len(self._keys)}), "
                    f"resetting to 0"
                )
                active_index = 0
                state['active_key_index'] = 0
                self._save_state(state)
            
            active_key = self._keys[active_index]
            logger.debug(
                f"Active OpenRouter key: {self._mask_key(active_key)} "
                f"(index {active_index + 1}/{len(self._keys)})"
            )
            return active_key
    
    def rotate_key_on_exhaustion(self, exhausted_key: str) -> Optional[str]:
        """
        Move to the next key when current key is exhausted.
        
        Args:
            exhausted_key: The API key that returned a 429 error
            
        Returns:
            The new active key, or None if no keys remain
        """
        with self._lock:
            state = self._load_state()
            active_index = state['active_key_index']
            
            # Verify the exhausted key matches current active key
            if active_index < len(self._keys) and self._keys[active_index] != exhausted_key:
                logger.warning(
                    f"Exhausted key doesn't match current active key at index {active_index}"
                )
            
            # Move to next key
            new_index = active_index + 1
            
            # Check if any keys remain
            if new_index >= len(self._keys):
                logger.error(
                    f"❌ All {len(self._keys)} OpenRouter API keys have been exhausted!"
                )
                return None
            
            # Update state
            state['active_key_index'] = new_index
            self._save_state(state)
            
            new_active_key = self._keys[new_index]
            logger.warning(
                f"⚠️  OpenRouter key EXHAUSTED: {self._mask_key(exhausted_key)} | "
                f"Rotated to key #{new_index + 1}/{len(self._keys)}: {self._mask_key(new_active_key)}"
            )
            
            return new_active_key
    
    def get_stats(self) -> dict:
        """
        Get statistics about the key manager.
        
        Returns:
            Dictionary with stats
        """
        with self._lock:
            state = self._load_state()
            active_index = state['active_key_index']
            
            keys_remaining = len(self._keys) - active_index
            
            return {
                "total_keys": len(self._keys),
                "active_key_index": active_index,
                "active_key_masked": self._mask_key(self._keys[active_index]) if active_index < len(self._keys) else "None",
                "keys_remaining": keys_remaining,
                "keys_exhausted": active_index
            }
    
    def reset_index(self) -> None:
        """
        Reset the active key index to 0 (start from first key again).
        Useful when you want to reuse all keys.
        """
        with self._lock:
            state = {"active_key_index": 0}
            self._save_state(state)
            logger.info(f"Key index reset to 0. All {len(self._keys)} keys available again.")
    
    @staticmethod
    def _mask_key(api_key: str) -> str:
        """
        Mask an API key for logging (show first 15 and last 10 chars).
        
        Args:
            api_key: The API key to mask
            
        Returns:
            Masked key string
        """
        if len(api_key) <= 25:
            return api_key[:10] + "..." + api_key[-5:]
        return api_key[:15] + "..." + api_key[-10:]


# Global singleton instance
_openrouter_key_manager: Optional[OpenRouterKeyManager] = None


def get_openrouter_key_manager(keys_from_env: Optional[List[str]] = None) -> OpenRouterKeyManager:
    """
    Get the global OpenRouterKeyManager instance (singleton pattern).
    
    Args:
        keys_from_env: List of API keys from .env (required on first call)
    
    Returns:
        OpenRouterKeyManager instance
    """
    global _openrouter_key_manager
    
    if _openrouter_key_manager is None:
        if keys_from_env is None:
            raise ValueError("keys_from_env required for first initialization")
        _openrouter_key_manager = OpenRouterKeyManager(keys_from_env)
    
    return _openrouter_key_manager
