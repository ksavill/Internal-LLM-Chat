# FILE: .dockerignore

# Git specific
.git
.gitignore

# Docker specific
Dockerfile
DockerFile.dockerfile
docker-compose.yml

# Python specific
*.pyc
__pycache__/
venv/
*.egg-info/
.env

# Application specific
repos_reference/ # Crucial: prevents copying local repo clones into the image

# OS / Editor specific
.DS_Store
.vscode/
*.swp