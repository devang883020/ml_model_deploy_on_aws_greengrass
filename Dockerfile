FROM python:3.9-slim

# Install requirements
RUN pip install --no-cache-dir scikit-learn numpy

# Copy files
COPY inference.py .
COPY iris_model.pkl .

RUN pip install --no-cache-dir numpy scikit-learn

CMD ["python3", "inference.py"]



