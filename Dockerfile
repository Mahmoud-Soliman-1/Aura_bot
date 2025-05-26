# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application files to the container
COPY . /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 7860

# Set environment variables for Hugging Face Spaces
ENV PORT=7860
ENV HOST=0.0.0.0

# Install Gunicorn (Production WSGI Server)
RUN pip install gunicorn

# Run the Flask app with Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:app"]