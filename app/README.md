# App

Streamlit web UI for natural language NAIP satellite image search using SkyCLIP embeddings and BigQuery vector search.

## How to Run

```bash
cd SkyScript
streamlit run app/streamlit_app.py
```

## Prerequisites

- **Model checkpoint**: SkyCLIP ViT-L-14 (~4.6 GB) at the path specified by `CKPT_PATH` in `.env`
- **GCP auth**: `gcloud auth application-default login`
- **BigQuery table**: Embeddings table with vector index (created by `embeddings/02-naip-embedding-pipeline.ipynb`)
- **GCS bucket**: NAIP tile images accessible to display in results

## Features

- **Text search**: Type a natural language query (e.g., "basketball court", "swimming pools") to find matching satellite tiles
- **Example chips**: Click pre-defined queries for quick demo
- **Image gallery**: Grid view of matching NAIP tiles with similarity scores and coordinates
- **Map view**: Interactive folium map with satellite/street/dark basemaps, color-coded markers by rank
- **Details table**: Full metadata with CSV export
- **Configurable**: Adjust result count, distance threshold, and search precision via sidebar

## Configuration

Copy `../.env.example` to `../.env` and fill in your values. The app uses:

- `PROJECT_NAME` — GCP project ID
- `DATASET_ID` — BigQuery dataset
- `TABLE_ID` — BigQuery table with embeddings
- `BUCKET_NAME` — GCS bucket with NAIP tile images
- `CKPT_PATH` — Path to SkyCLIP model checkpoint

## Deployment (Cloud Run)

The app deploys to Google Cloud Run for demo use. `deploy.sh` and `Dockerfile` are at the repo root (Cloud Build context).

**Prerequisites:**
- GCP project with billing enabled
- `gcloud` CLI authenticated (`gcloud auth login`)
- `.env` with `PROJECT_NAME`, `BUCKET_NAME`, `DATASET_ID`, `TABLE_ID` set (all required, no defaults)

**Commands:**

```bash
# Deploy to Cloud Run (builds image, pushes to Artifact Registry, deploys)
./deploy.sh deploy

# Check service status and URL
./deploy.sh status

# Tail logs
./deploy.sh logs

# Tear down when done (deletes the Cloud Run service, keeps image for fast redeploy)
./deploy.sh stop

# Full cleanup (deletes service + Artifact Registry images)
./deploy.sh clean
```

**Notes:**
- First request takes 2-3 minutes (model loads into memory on cold start)
- Scales to zero when idle — no cost when not in use (`min-instances 0`)
- The 4.6GB model checkpoint is baked into the Docker image at build time
- Cloud Run config: 32GB RAM, 8 CPUs, concurrency 80, max 3 instances
- `stop` keeps the image in Artifact Registry (~$0.70/month) so next deploy is fast
- `clean` removes everything including the image (next deploy requires full rebuild)
