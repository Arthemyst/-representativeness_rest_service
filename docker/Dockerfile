# Dockerfile

# Use the official Python 3.9 base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code to the working directory
COPY . .

# Create the cache directory with appropriate permissions
RUN mkdir -p /app/.pytest_cache/v/cache

# Set the command to run when the container starts
CMD ["python", "flask_app.py"]
