FROM python:3.11-slim

LABEL maintainer="Nikhil Reddy Donapati <nikhil.donapati@myemail.indwes.edu>"
LABEL description="LLM-FMEA Supply Chain Risk Assessment — API + Dashboard"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/nikhildonapati/llm-fmea-scra"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed data/synthetic results/figures logs

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# API port
EXPOSE 8000
# Streamlit port
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Default: API server (override with streamlit run demo/dashboard.py for demo)
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
