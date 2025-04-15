# Model Aliases Configuration

The Internal LLM Chat supports model aliases, which allow you to define friendly names for LLM models and use them interchangeably with the actual model names.

## Configuration Files

Model aliases are configured using JSON files in the `model_aliases` directory. The system determines which file to use based on the `NODE_ENV` environment variable:

- Set `NODE_ENV` to `"production"` to use settings from `model_aliases/production.json`.
- Set `NODE_ENV` to `"development"` to use settings from `model_aliases/development.json`.
- If `NODE_ENV` is not set or is empty, the application loads `model_aliases/no_environment.json`.

## File Format

Each model aliases file should contain a JSON object mapping alias names to actual model names:

```json
{
    "alias1": "actual-model-name-1",
    "alias2": "actual-model-name-2"
}
```

## Example

Consider this example configuration in `model_aliases/any_environment.json`:

```json
{
    "gpt4": "gpt-4o",
    "gpt3": "gpt-3.5-turbo",
    "llama": "llama3:8b",
    "fast": "qwen2.5-coder:7b",
    "pro": "o1-pro"
}
```

With this configuration, users can use "gpt4" instead of "gpt-4o" when making requests, and the system will automatically use "gpt-4o".

## Environment-Specific Configurations

You can create environment-specific alias configurations:

- `model_aliases/development.json` for development environments
- `model_aliases/production.json` for production environments
- `model_aliases/staging.json` for staging environments

The system will load the appropriate file based on the `NODE_ENV` environment variable.

## Usage

To use model aliases:

1. Define your aliases in the appropriate JSON file in the `model_aliases` directory.
2. Set the `NODE_ENV` environment variable to specify which environment you're in (optional).
3. Use the alias names in your requests to the Internal LLM Chat.

The system will automatically resolve the aliases to the corresponding actual model names.

## Implementation Details

The model alias resolution happens transparently in the backend:

1. When the server starts, it loads the appropriate model aliases file based on the `NODE_ENV` environment variable.
2. When a request is made with a model name, the system checks if it's an alias.
3. If it's an alias, the system replaces it with the actual model name before processing the request.
4. The response is returned as if the actual model name was used directly.

This allows for consistent model references across different environments without changing the client code.