FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install headless OpenCV FIRST, then remaining deps.
# ultralytics depends on opencv-python, but pip will see cv2 is
# already satisfied by the headless variant and skip it.
COPY requirements.txt .
RUN pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
