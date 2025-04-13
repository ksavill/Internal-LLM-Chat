import os
import json
import logging
from typing import Dict, Any, Optional
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)

def _get_profiles_file_path() -> Path:
    """
    Returns the JSON file path based on the NODE_ENV environment variable.
    """
    env = os.environ.get("NODE_ENV")
    profiles_dir = Path("request_profiles")
    file_path = profiles_dir / f"{env}.json" if env else profiles_dir / "no_environment.json"
    logger.debug(f"Determined profiles file path: {file_path}")
    return file_path

async def load_profile(profile_name: str) -> Optional[Dict[str, Any]]:
    """
    Load a profile configuration from the appropriate environment JSON file.

    Args:
        profile_name: The name of the profile to load.

    Returns:
        A dictionary containing the profile configuration or None if not found.
    """
    file_path = _get_profiles_file_path()

    if not file_path.exists():
        logger.warning(f"Profiles file {file_path} does not exist.")
        return None

    try:
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
            profiles = json.loads(content)
            logger.debug(f"Loaded profiles from {file_path}: {list(profiles.keys())}")
            return profiles.get(profile_name)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading profile from {file_path}: {e}")
        return None

async def load_all_profiles() -> Dict[str, dict]:
    """
    Load all request profiles from the environment-based JSON file.
    Returns a dictionary mapping profile_name -> {model, backup_models, ...}.
    """
    file_path = _get_profiles_file_path()

    if not file_path.exists():
        logger.warning(f"Profiles file {file_path} does not exist. Returning an empty profiles dictionary.")
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Loaded all profiles from {file_path}: {list(data.keys())}")
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Error loading all profiles from {file_path}: {e}")
        return {}