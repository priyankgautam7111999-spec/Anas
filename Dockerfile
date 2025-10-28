# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY bot_super_advanced.py .

# Set environment variables (override with actual values at runtime)
ENV PYTHONUNBUFFERED=1

# Run the bot
CMD ["python", "bot_super_advanced.py"]
