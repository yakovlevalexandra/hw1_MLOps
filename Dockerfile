# # For more information, please refer to https://aka.ms/vscode-docker-python
# FROM python:3.8-slim

# EXPOSE 5002

# # Keeps Python from generating .pyc files in the container
# ENV PYTHONDONTWRITEBYTECODE=1

# # Turns off buffering for easier container logging
# ENV PYTHONUNBUFFERED=1

# # Install pip requirements
# COPY requirements.txt .
# RUN python -m pip install -r requirements.txt

# WORKDIR /app
# COPY . /app

# # Creates a non-root user with an explicit UID and adds permission to access the /app folder
# # For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# # During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# # CMD ["gunicorn", "--bind", "0.0.0.0:5002", "code\api:app"]
# CMD python code\api.py:app

FROM python:3.8-slim

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# WORKDIR /code

EXPOSE 5000

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . .

CMD python code/api.py





# FROM python:3.9-slim

# RUN apt-get update \
# && apt-get install gcc -y \
# && apt-get clean

# ENV POETRY_HOME="/opt/poetry" \
#     POETRY_VIRTUALENVS_CREATE=false \
#     POETRY_VIRTUALENVS_IN_PROJECT=false \
#     POETRY_NO_INTERACTION=1 \
#     POETRY_VERSION=1.1.14

# RUN pip install --user poetry
# ENV PATH="${PATH}:/root/.local/bin"

# WORKDIR /opt/ml_service

# COPY poetry.lock pyproject.toml /

# RUN poetry config virtualenvs.create false && \
#     poetry install --no-interaction --no-ansi --no-root

# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# COPY . .

# CMD ["python", "main.py"]