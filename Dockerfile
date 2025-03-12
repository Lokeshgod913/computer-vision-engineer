# Use official Python image as base
FROM python:3.8

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
