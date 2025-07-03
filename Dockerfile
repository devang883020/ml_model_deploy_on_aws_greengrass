FROM python:3.9-slim

# Install requirements
RUN pip install --no-cache-dir scikit-learn numpy

# Copy files
COPY iris_model.pkl /app/
COPY inference.py /app/

# Set working directory
WORKDIR /app

# Run inference
CMD ["python", "inference.py"]


