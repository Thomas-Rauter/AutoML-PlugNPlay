# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Input is mounted here.
RUN mkdir /app/input

# Output is copied from here.
RUN mkdir /app/output

# Copy the python script Optuna_CatBoost.py into the container at /app
COPY ./Optuna_CatBoost.py /app

# Install required packages
RUN pip install --no-cache-dir catboost optuna pandas matplotlib seaborn reportlab scikit-learn

# Run Optuna_CatBoost.py when the container launches
ENTRYPOINT ["python", "./Optuna_CatBoost.py"]








# Put this file into a dir with only this file and the python script, open a terminal and run this command (without the "): "docker build -t thomasrauter/autocatboost:1.0 ."
