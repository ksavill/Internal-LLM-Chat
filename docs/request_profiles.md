# Request Profiles Configuration

## Overview

The Internal LLM Chat supports request profiles that allow you to override the model and backup models specified in a request. This document explains how to configure and use request profiles.

## How It Works

1. When a request includes a `request_profile` field, the system will look for that profile in the environment-specific JSON files.
2. If the profile is found, the `model` and `backup_models` specified in the request will be overridden with the values from the profile.
3. If the profile cannot be found, the request will receive a 400 error with a message indicating an invalid request profile.

## Environment Configuration

The system uses the `NODE_ENV` environment variable to determine which profile configuration file to use:

- If `NODE_ENV` is set to 'development', the system will use `request_profiles/development.json`.
- If `NODE_ENV` is set to 'production', the system will use request_profiles/production.json`.
- If `NODE_ENV` is set to 'test', the system will use `request_profiles/test.json`.
- If `NODE_ENV` is not set, the system will use `request_profiles/no_environment.json`.

## Profile Configuration Files

Profile configuration files are JSON files stored in the `request_profiles/` directory. Each file contains a mapping of profile names to their model configurations:

```json
{
  "default_profile": {
    "model": "gpt-4o",
    "backup_models": ["gpt-3.5-turbo", "qwen2.5-coder:7b"]
  },
  "fast_profile": {
    "model": "gpt-3.5-turbo",
    "backup_models": ["qwen2.5-coder:7b"]
  }
}
```

## Usage Example

To use a request profile in your API request:

```json
{
  "request_profile": "fast_profile",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ]
}
```

This will override any `model` or `backup_models` specified in the request with the values from the 'fast_profile' configuration.

## Adding New Profiles

To add a new profile:

1. Identify the appropriate environment file in the `request_profiles/` directory.
2. Add your new profile to the JSON file with the desired model and backup models.
3. Restart the server to ensure the new profile is loaded.

## Default Profiles

Each environment file should include a 'default_profile' that will be used if no profile is specified in the request.