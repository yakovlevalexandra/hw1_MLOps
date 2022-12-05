FROM python:3.8-slim

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# WORKDIR /code

EXPOSE 5000

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . .

CMD python code/api.py
