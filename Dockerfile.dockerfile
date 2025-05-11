# FILE: DockerFile.dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.9-slim

# 2. Install git (needed for cloning repositories)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory in the container
WORKDIR /app

# 4. Copy the requirements file into the container
COPY requirements.txt .

# 5. Install any needed packages specified in requirements.txt
#    Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application's code into the container
#    (main.py, index.html).
#    Ensure .dockerignore is present to avoid copying unnecessary files (like local repos_reference).
COPY . .

# 7. Make port 32546 available to services outside this container
EXPOSE 32546

# 8. Define the command to run your application
#    This will execute: uvicorn main:app --host 0.0.0.0 --port 32546
#    The main.py script will create /app/repos_reference if it doesn't exist on first run
#    before volume mount populates it.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "32546"]