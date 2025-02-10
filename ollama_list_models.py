"""
This script calls "ollama list" via subprocess,
parses its output into a structured list of dictionaries,
and prints the available models in a clean format.

Assumptions about the CLI output:
  • The output contains a header row with at least the following
    columns: NAME, ID, SIZE, MODIFIED.
  • Each subsequent non-empty line is a row with the data for one model.
  • The ID is assumed to be a 12-character hexadecimal string.
  • The SIZE column is assumed to be two tokens (e.g., '9.1 GB').
  • The MODIFIED column may consist of multiple tokens (e.g., '21 hours ago').
"""

import subprocess
import re
import sys

def is_model_id(token):
    """
    Determines if a token is likely a model ID.
    We expect a 12-character hexadecimal string, e.g., "ac896e5b8b34".
    """
    return bool(re.fullmatch(r"[0-9a-f]{12}", token))

def parse_row(row, header_keys):
    """
    Parse one row of text into a dictionary using header_keys.
    First attempts to split the row using two or more spaces. If that does
    not yield the expected number of columns, it splits by whitespace and 
    heuristically identifies the boundaries.
    """
    # Try splitting on two or more spaces first.
    fields = re.split(r"\s{2,}", row.strip())
    if len(fields) == len(header_keys):
        return dict(zip(header_keys, fields))
    else:
        # Fallback: split all whitespace.
        tokens = row.strip().split()
        # Locate the token that matches the ID pattern.
        id_index = None
        for idx, token in enumerate(tokens):
            if is_model_id(token):
                id_index = idx
                break
        if id_index is None:
            raise ValueError("Could not locate a valid model ID in row: " + row)
        # The model NAME is defined as all tokens before the model ID.
        name = " ".join(tokens[:id_index])
        model_id = tokens[id_index]
        # We expect the SIZE to be the next two tokens (e.g., "9.1 GB").
        if len(tokens) <= id_index + 1:
            raise ValueError("Not enough tokens after model ID for SIZE in row: " + row)
        size = " ".join(tokens[id_index+1:id_index+3])
        # The remaining tokens make up the MODIFIED field.
        modified = " ".join(tokens[id_index+3:])
        return dict(zip(header_keys, [name, model_id, size, modified]))

def parse_models(raw_lines):
    """
    Given the raw lines from the CLI output, identify the header row and parse
    each subsequent row into a dictionary using the header tokens.
    """
    header_line = None
    for line in raw_lines:
        if "NAME" in line and "ID" in line and "SIZE" in line and "MODIFIED" in line:
            header_line = line
            break

    if header_line is None:
        raise ValueError("Header row not found in output.")

    # Try to extract header tokens using multiple-space separation; if not,
    # fall back to splitting on any whitespace.
    header_tokens = re.split(r"\s{2,}", header_line.strip())
    if len(header_tokens) < 4:
        header_tokens = header_line.strip().split()

    header_keys = header_tokens[:4]  # Expecting NAME, ID, SIZE, MODIFIED

    # Parse each line following the header.
    models = []
    start_parsing = False
    for line in raw_lines:
        if not start_parsing:
            if line == header_line:
                start_parsing = True
            continue
        if not line.strip():
            continue
        try:
            model = parse_row(line, header_keys)
            models.append(model)
        except Exception as e:
            print("Error parsing row:", line, "\nError:", e, file=sys.stderr)
    return models

def list_models():
    """
    Calls "ollama list" using subprocess, splits its output by lines,
    and returns a structured list of dictionaries for the models.
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing 'ollama list':", e.stderr, file=sys.stderr)
        raise e

    output = result.stdout.strip()
    if not output:
        print("No output received from 'ollama list'.", file=sys.stderr)
        return []

    raw_lines = output.splitlines()
    return parse_models(raw_lines)

if __name__ == "__main__":
    try:
        models = list_models()
        print("Available models:")
        for model in models:
            print(model)
    except Exception as err:
        print("An error occurred:", err, file=sys.stderr)
        sys.exit(1)