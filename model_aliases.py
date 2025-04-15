import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import aiofiles

logger = logging.getLogger(__name__)

MODEL_ALIASES_DIR = Path("model_aliases")
# CACHE maps file path (string) to a tuple containing (cache_time, alias_dict)
CACHE: Dict[str, Tuple[datetime, Dict[str, str]]] = {}

async def get_model_aliases() -> Dict[str, str]:
    """
    Asynchronously load the model aliases for the current environment.

    Returns:
        dict: A dictionary mapping model aliases to their target models.
              Returns an empty dictionary if no alias file is found or if an error occurs.
    """
    env = os.environ.get("NODE_ENV", "").strip()
    if env:
        file_path = MODEL_ALIASES_DIR / f"{env}.json"
    else:
        file_path = MODEL_ALIASES_DIR / "no_environment.json"
    file_path_str = str(file_path)

    if not file_path.exists():
        logger.warning(f"Alias file {file_path} does not exist. Returning empty aliases mapping.")
        return {}

    now = datetime.now()
    # Check the cache first.
    if file_path_str in CACHE:
        cache_time, cached_aliases = CACHE[file_path_str]
        if (now - cache_time).total_seconds() < 60:
            logger.debug(f"Using cached aliases from {file_path} (cached at {cache_time}).")
            return cached_aliases
        else:
            logger.debug(f"Cache expired for {file_path} (cached at {cache_time}). Reloading aliases.")
    else:
        logger.debug(f"No cache available for {file_path}. Loading aliases from file.")

    try:
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
            data = json.loads(content)
            # Update cache with current timestamp.
            CACHE[file_path_str] = (now, data)
            logger.debug(f"Loaded aliases from {file_path}: {data}")
            return data
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading {file_path}: {e}")
        return {}

async def resolve_model_alias(model_name: str) -> str:
    """
    Asynchronously resolve a model name against the configured aliases.
    
    Args:
        model_name (str): The model name to resolve.
        
    Returns:
        str: The resolved model name if an alias is found; otherwise the original model name.
    """
    aliases = await get_model_aliases()
    if model_name in aliases:
        resolved_name = aliases[model_name]
        logger.debug(f"Resolved model alias: {model_name} -> {resolved_name}")
        return resolved_name
    else:
        logger.debug(f"No alias found for model '{model_name}'. Returning the original name.")
        return model_name
