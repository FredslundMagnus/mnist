# Base image
FROM python:3.11.1-slim-buster

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy data
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/

# Set working directory
WORKDIR /
RUN mkdir -p /reports
RUN mkdir -p /reports/figures

# Install requirements
RUN pip install -r requirements.txt --no-cache-dir

# Set entrypoint
ENTRYPOINT ["python", "-u", "src/models/predict_model.py", "predict", "models/trained_model.pt"]