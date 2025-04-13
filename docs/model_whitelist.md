# Model Whitelisting Setup

This guide explains how to configure whitelists for allowed models based on the environment your application is running in.

## Overview

Your application controls which models can be used through a whitelist mechanism. The whitelist is determined by:

- An **environment variable** that specifies the current environment.
- A corresponding **JSON file** that lists the allowed models.

## Environment Variable

The application reads the `NODE_ENV` environment variable to determine which whitelist file to load. For example:

- Set `NODE_ENV` to `"production"` to use settings from `model_whitelists/production.json`.
- Set `NODE_ENV` to `"development"` to use settings from `model_whitelists/development.json`.
- If `NODE_ENV` is not set or is empty, the application loads `model_whitelists/no_environment.json`.

### How to Set `NODE_ENV`

**On Linux/Mac:**

```bash
export NODE_ENV=production
On Windows (PowerShell):

powershell
Copy
$env:NODE_ENV="production"
Make sure to set the environment variable before running your application.

JSON Whitelist File Format
Whitelist files should be stored in the model_whitelists directory. The file name must match the value of NODE_ENV, with the .json extension. If no environment is set, use no_environment.json.

JSON Structure
Each JSON file should follow this simple format:

json
{
    "allowed_models": [
        "gpt-4*",
        "gpt-3.5*",
        "custom-model"
    ]
}
Key: allowed_models

Value: An array of model patterns.

The patterns can include wildcards (using Unix shell-style matching) to provide flexibility.

For example, "gpt-4*" allows any model name that starts with "gpt-4".

Example Setup
Create the Whitelist Directory:
Ensure that a directory named model_whitelists exists in your project root.

Create a JSON File:
For production, create a file named production.json with content like:

json
{
    "allowed_models": [
        "gpt-4*",
        "gpt-3.5*"
    ]
}
Set the Environment Variable:
Set NODE_ENV to "production" as shown in the previous section.

Restart Your Application:
Once the environment variable is set and the JSON file is in place, restart your application. It will load the whitelist from model_whitelists/production.json and restrict models to those matching the provided patterns.

How It Works in Code
Loading the Whitelist:
The application checks the NODE_ENV environment variable. If set, it looks for a JSON file named <NODE_ENV>.json in the model_whitelists directory. If not set, it defaults to no_environment.json.

Allowed Models Verification:
The function (e.g., is_model_allowed) reads the allowed_models list and uses Unix shell-style matching (using the fnmatch module) to determine if a given model name is permitted.

Fallback Behavior:
If the whitelist file is missing or there’s a parsing error, the function returns True, allowing all models by default. This provides a safe fallback if configuration issues arise.