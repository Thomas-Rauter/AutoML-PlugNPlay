# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Create the directory structure inside the Docker container.
RUN mkdir /main
RUN mkdir /main/input
RUN mkdir /main/scripts
RUN mkdir /main/app
RUN mkdir /main/output

# Set the working directory in the container
WORKDIR /main/app

# Copy the current directory contents into the container at /app
COPY ./nn_hyperpara_screener.py /main/scripts

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir pandas torch matplotlib reportlab numpy optuna plotly

# So that matplotlib uses /tmp/matplotlib as config dir, which is  writable by any user.
ENV MPLCONFIGDIR=/tmp/matplotlib

# Run nn_hyperpara_screener.py when the container launches
ENTRYPOINT ["python", "../scripts/nn_hyperpara_screener_for_container.py"]





# docker build -t nn_hyperpara_screener -f Dockerfile .
