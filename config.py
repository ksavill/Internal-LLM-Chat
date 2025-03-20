import json
import aiofiles
from fastapi import FastAPI
import logging

logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"

async def load_config(app: FastAPI):
    """Load the configuration from config.json into app.state.config."""
    try:
        async with aiofiles.open(CONFIG_FILE, "r") as f:
            content = await f.read()
            app.state.config = json.loads(content)
    except FileNotFoundError:
        app.state.config = {}
    except json.JSONDecodeError:
        logger.error("Invalid config.json format")
        app.state.config = {}

async def save_config(app: FastAPI):
    """Save the current app.state.config to config.json."""
    async with aiofiles.open(CONFIG_FILE, "w") as f:
        await f.write(json.dumps(app.state.config, indent=4))

async def get_value(app: FastAPI, key: str):
    """
    Asynchronously retrieve a value from app.state.config using a dot-separated key.
    Returns None if the key doesn't exist.
    """
    keys = key.split(".")
    return get_nested_value(app.state.config, keys)

async def set_value(app: FastAPI, key: str, value):
    """
    Asynchronously set a value in app.state.config using a dot-separated key and save to config.json.
    Creates intermediate dictionaries if they don't exist.
    """
    async with app.state.config_lock:
        keys = key.split(".")
        set_nested_value(app.state.config, keys, value)
        await save_config(app)

def get_nested_value(d, keys):
    """Helper function to get a nested value from a dictionary."""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return None
    return d

def set_nested_value(d, keys, value):
    """Helper function to set a nested value in a dictionary."""
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value