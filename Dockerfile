# Stage 1: Use an official Python runtime as a parent image
# We choose a specific version for reproducibility. 'slim' is a smaller version.
FROM python:3.9-slim

# Set the working directory inside the container
# All subsequent commands will be run from this directory
WORKDIR /app

# Copy the file that lists our dependencies into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir makes the image smaller
# --trusted-host is sometimes needed to avoid SSL issues in certain networks
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of our application's code and assets into the container
# This includes app.py, the model, and the char map
COPY app/ .

# Tell Docker that the container listens on port 5000 at runtime
# This is for documentation and for other containers to know how to connect
EXPOSE 5000

# Define the command to run your application
# This is what starts the Flask server when a container is created from the image
CMD ["python", "app.py"]
