FROM python:3.11-slim

WORKDIR /app

# Install system-level dependencies required by Prophet & NumPy
RUN apt-get update && apt-get install -y \
    build-essential \
    libpython3-dev \
    libatlas-base-dev \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt


# Copy the rest of the app
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI app
CMD ["python", "run_server.py"]
