import os
import json
import logging
import fnmatch
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

WHITELIST_DIR = Path("model_whitelists")
# Cache maps absolute file path (as a string) to a tuple (last_load_time, whitelist_value)
CACHE = {}

def get_model_whitelist():
    """
    Get the model whitelist for the current environment.

    This function caches whitelist contents and refreshes the cache if the
    cache is older than 60 seconds.

    Returns:
        list: List of allowed model patterns, or None if no restrictions.
    """
    env = os.environ.get("NODE_ENV", "").strip()
    file_path = WHITELIST_DIR / (f"{env}.json" if env else "no_environment.json")
    file_path_str = str(file_path)

    if not file_path.exists():
        logger.warning(f"Whitelist file {file_path} does not exist. No restrictions applied.")
        return None

    now = datetime.now()
    # Check cache for the file path
    if file_path_str in CACHE:
        cache_time, cached_whitelist = CACHE[file_path_str]
        if (now - cache_time).total_seconds() < 60:
            logger.debug(f"Using cached whitelist for {file_path}. Cache timestamp: {cache_time}")
            return cached_whitelist
        else:
            logger.debug(f"Cache expired for {file_path} (cached at {cache_time}). Reloading whitelist.")
    else:
        logger.debug(f"No cache found for {file_path}. Loading whitelist from file.")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        whitelist = data.get("allowed_models", [])
        CACHE[file_path_str] = (now, whitelist)
        logger.debug(f"Loaded whitelist from {file_path}: {whitelist}")
        return whitelist
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def is_model_allowed(model_name):
    """
    Check if a model is allowed based on the current environment's whitelist.

    Args:
        model_name (str): Name of the model to check

    Returns:
        bool: True if model is allowed, False otherwise.
    """
    whitelist = get_model_whitelist()

    if whitelist is None:
        logger.info("No whitelist applied (returned None). Allowing all models.")
        return True

    for pattern in whitelist:
        if fnmatch.fnmatch(model_name, pattern):
            logger.debug(f"Model '{model_name}' matches allowed pattern '{pattern}'.")
            return True

    logger.debug(f"Model '{model_name}' does not match any allowed patterns in whitelist: {whitelist}")
    return False