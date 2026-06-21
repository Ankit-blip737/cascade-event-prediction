# CASCADE dashboard — containerized for one-URL deploy (Render / Railway / Fly / any Docker host).
FROM python:3.12-slim

WORKDIR /app

# system deps (pyarrow/ortools wheels are self-contained; just need build-less runtime)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# app code + precomputed artifacts (small parquet/npz/json — well under platform limits)
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY models/ ./models/
COPY .streamlit/ ./.streamlit/

EXPOSE 7860
HEALTHCHECK CMD curl -fsS http://localhost:7860/_stcore/health || exit 1

# $PORT is set by most PaaS; default to 7860 for Hugging Face Spaces
CMD streamlit run src/cascade/demo/app.py --server.port ${PORT:-7860} --server.address 0.0.0.0 --server.headless true
