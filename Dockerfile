
# Use the official Python image as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set the entry point for the container
CMD ["python", "main.py"]
