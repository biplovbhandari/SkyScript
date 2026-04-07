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
