# Embeddings

Generate SkyCLIP embeddings from satellite imagery and load them into BigQuery for vector search.

## Notebooks

### `01-local-embedding-search.ipynb`
Local proof-of-concept for text-to-image search.

1. Loads the SkyCLIP model (ViT-L-14, 768-dim embeddings)
2. Encodes images from the SkyScript test CSV (30K image-text pairs)
3. Builds a FAISS index (inner product on L2-normalized vectors = cosine similarity)
4. Runs text queries ("airplanes", "basketball court", etc.) and visualizes matches

### `02-naip-embedding-pipeline.ipynb`
End-to-end production pipeline that feeds the Streamlit search app.

1. **Configuration** — Loads GCP project, bucket, and region from `.env`
2. **Download NAIP Tiles from Earth Engine** — Grids the region into 256m tiles, filters to valid (non-empty) tiles, exports to GCS
3. **Compute SkyCLIP Embeddings** — Downloads each tile from GCS, extracts metadata (bounds, CRS, acquisition time), transforms coordinates to WGS84, computes 768-dim embedding
4. **Load Embeddings into BigQuery** — Creates table with embedding, bbox, geometry, and acquisition time columns
5. **Test: Text-to-Tile with ML Distance** — Baseline search using raw ML distance (poor results)
6. **Test: Local Search with FAISS** — Downloads embeddings from BigQuery, builds local FAISS index
7. **Test: BigQuery Vector Search** — Uses `VECTOR_SEARCH` with IVF index (1000 lists, cosine distance) — this is the approach the Streamlit app uses

## Other Files

- `skyscript_img_embeddings.npz` — Cached embeddings from `01-local-embedding-search.ipynb`

## Prerequisites

- GCP project with BigQuery and Cloud Storage enabled
- Earth Engine access (`ee.Initialize()`)
- SkyCLIP model checkpoint (see `.env.example` for path)
- Python dependencies: `uv sync` from the repo root

## Configuration

Copy `../.env.example` to `../.env` and fill in your values. See the example file for all required variables.
