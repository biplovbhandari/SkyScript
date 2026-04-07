FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including GDAL for rasterio
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    unzip \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config

# Copy uv files first (for better caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen

# Download and extract the model checkpoint
RUN mkdir -p data/checkpoints && \
    curl -o model.zip https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS.zip && \
    unzip model.zip -d data/checkpoints/ && \
    rm model.zip

# Copy project files (.dockerignore excludes dev files)
COPY . .

ENV PYTHONPATH="${PYTHONPATH}:/app/src:/app"

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

CMD uv run streamlit run app/streamlit_app.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
