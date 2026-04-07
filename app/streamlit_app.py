# streamlit_vector_search.py

import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import torch
import numpy as np
from google.cloud import bigquery, storage
from PIL import Image
import io
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime
import time
import types

from src.open_clip.factory import create_model_and_transforms, get_tokenizer

# Page config
st.set_page_config(
    page_title="Visual Language Model with NAIP Satellite Image Search",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# Configuration
PROJECT_NAME = os.getenv("PROJECT_NAME", "your-gcp-project")
DATASET_ID = os.getenv("DATASET_ID", "your-bq-dataset")
TABLE_ID = os.getenv("TABLE_ID", "your-bq-table")
BUCKET_NAME = os.getenv("BUCKET_NAME", "your-gcs-bucket")
precision="amp"
ckpt_name = os.getenv("CKPT_PATH", "data/checkpoints/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt")
model_arch_name = "ViT-L-14"


@st.cache_resource
def get_storage_client():
    """Singleton GCS client, shared across all image loads."""
    return storage.Client(project=PROJECT_NAME)


@st.cache_resource
def load_model():
    """Load the model and tokenizer (cached)."""
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    try:
        status_placeholder.info("Checking checkpoint file...")
        progress_bar.progress(10)

        if not os.path.exists(ckpt_name):
            st.error(f"Checkpoint file not found at: {ckpt_name}")
            return None, None, None

        checkpoint_size = os.path.getsize(ckpt_name) / (1024*1024*1024)
        status_placeholder.info(f"Checkpoint found ({checkpoint_size:.2f} GB)")
        progress_bar.progress(20)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        status_placeholder.info(f"Using device: {device}")
        progress_bar.progress(40)

        status_placeholder.info("Loading model checkpoint... (this may take 2-3 minutes)")
        progress_bar.progress(50)

        start_time = time.time()
        model, _, _ = create_model_and_transforms(
            model_arch_name,
            ckpt_name,
            precision=precision,
            device=device,
            output_dict=True,
            force_quick_gelu=True,
        )
        load_time = time.time() - start_time
        status_placeholder.info(f"Model loaded in {load_time:.1f} seconds")
        progress_bar.progress(80)

        status_placeholder.info("Loading tokenizer...")
        tokenizer = get_tokenizer(model_arch_name)
        progress_bar.progress(90)

        model.eval()

        status_placeholder.success("Model loaded successfully!")
        progress_bar.progress(100)

        time.sleep(1)
        status_placeholder.empty()
        progress_bar.empty()

        return model, tokenizer, device

    except Exception as e:
        status_placeholder.error(f"Error loading model: {str(e)}")
        progress_bar.empty()

        st.error("**Detailed Error Information:**")
        st.code(str(e))

        with st.expander("Debug Information"):
            st.write(f"Current working directory: {os.getcwd()}")
            st.write(f"Checkpoint path: {ckpt_name}")
            st.write(f"Checkpoint exists: {os.path.exists(ckpt_name)}")
            st.write(f"Python path: {sys.path[:3]}...")
            st.write(f"PyTorch version: {torch.__version__}")
            st.write(f"CUDA available: {torch.cuda.is_available()}")

        return None, None, None


def encode_text_query(query_text, model, tokenizer, device):
    """Encode text query to embedding vector using the model."""
    try:
        # Tokenize the text
        tokens = tokenizer([query_text])[0].to(device)

        with torch.no_grad():
            # Encode text
            query_vector = model.encode_text(tokens.unsqueeze(0), normalize=True).cpu().numpy()
        return query_vector
    except Exception as e:
        st.error(f"Error encoding text query: {e}")
        st.error(f"Query text: {query_text}")
        st.error(f"Tokens shape: {tokens.shape if 'tokens' in locals() else 'N/A'}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def vector_search_bq(query_vector_list, project_id, dataset_id, table_id,
                     top_k=20, distance_threshold=0.8, fraction_lists_to_search=0.01):
    """Perform vector search in BigQuery using VECTOR_SEARCH (cached)"""

    client = bigquery.Client(project=project_id)

    # Build the query
    query = f"""
    WITH query_embedding AS (
      SELECT {query_vector_list} AS embedding
    )

    SELECT
      base.tile_id,
      base.bbox_west_wgs84 AS bbox_west,
      base.bbox_south_wgs84 AS bbox_south,
      base.bbox_east_wgs84 AS bbox_east,
      base.bbox_north_wgs84 AS bbox_north,
      base.acquisition_time,
      distance
    FROM
      VECTOR_SEARCH(
        TABLE `{project_id}.{dataset_id}.{table_id}`,
        'embedding',
        TABLE query_embedding,
        TOP_K         => {top_k},
        DISTANCE_TYPE => 'COSINE',
        OPTIONS       => '{{"fraction_lists_to_search": {fraction_lists_to_search}}}'
      )
    WHERE distance > 0
      AND distance < {distance_threshold}
    ORDER BY distance
    LIMIT {top_k};
    """

    try:
        query_job = client.query(query)
        results = [types.SimpleNamespace(**dict(row)) for row in query_job.result()]
        return results
    except Exception as e:
        st.error(f"Error in BigQuery vector search: {e}")
        st.error(f"Query: {query[:200]}..." if len(query) > 200 else query)
        return []


def text_search_bq(query_text, model, tokenizer, device, project_id, dataset_id, table_id,
                   top_k=20, distance_threshold=0.8, fraction_lists_to_search=0.01):
    """Complete pipeline: text query -> embedding -> BigQuery vector search"""

    # Show progress
    with st.spinner("Encoding text query..."):
        query_vector = encode_text_query(query_text, model, tokenizer, device)

    if query_vector is None:
        return []

    # Convert to list for BigQuery
    if query_vector.ndim == 2:
        query_vector = query_vector.flatten()
    query_vector_list = query_vector.tolist()

    # Perform search
    with st.spinner("Searching satellite images..."):
        results = vector_search_bq(
            query_vector_list, project_id, dataset_id, table_id,
            top_k, distance_threshold, fraction_lists_to_search
        )

    return results


@st.cache_data(ttl=3600)
def load_image_from_gcs(gcs_path, bucket_name):
    """Load image from Google Cloud Storage (cached)."""
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        img_bytes = blob.download_as_bytes()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        st.error(f"Error loading image {gcs_path}: {e}")
        return None


def create_results_map(results):
    """Enhanced map creation with proper coordinate handling"""
    if not results:
        return None

    # Calculate center of all results (coordinates should already be transformed to WGS84)
    try:
        lats = [(r.bbox_south + r.bbox_north) / 2 for r in results]
        lons = [(r.bbox_west + r.bbox_east) / 2 for r in results]

        center_lat = np.mean(lats)
        center_lon = np.mean(lons)

        # Validate center coordinates
        if abs(center_lat) > 90 or abs(center_lon) > 180:
            st.error(f"❌ Invalid center coordinates: {center_lat:.6f}, {center_lon:.6f}")
            return None

    except Exception as e:
        st.error(f"Error calculating map center: {e}")
        return None

    # Create map with basic OpenStreetMap tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap"
    )

    # ONLY add tiles that work without attribution issues
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    # Add CartoDB tiles
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Light Map",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="Dark Map",
        overlay=False,
        control=True
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 10px; z-index: 1000;
                background-color: white; padding: 10px 14px; border-radius: 6px;
                border: 2px solid rgba(0,0,0,0.2); font-size: 13px;
                line-height: 1.8; box-shadow: 0 1px 5px rgba(0,0,0,0.3);">
        <b>Results</b><br>
        <i style="background:#cb3334; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:6px;"></i> Best match<br>
        <i style="background:#38aadd; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:6px;"></i> Top 3<br>
        <i style="background:#72af26; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:6px;"></i> Other matches
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add markers and rectangles for each result
    for i, result in enumerate(results):
        try:
            # Calculate center point
            lat = (result.bbox_south + result.bbox_north) / 2
            lon = (result.bbox_west + result.bbox_east) / 2

            # Validate coordinates
            if abs(lat) > 90 or abs(lon) > 180:
                st.warning(f"⚠️ Skipping result {i+1} - invalid coordinates: {lat:.6f}, {lon:.6f}")
                continue

            # Create popup content with more details
            bbox_width = abs(result.bbox_east - result.bbox_west)
            bbox_height = abs(result.bbox_north - result.bbox_south)

            popup_content = f"""
            <div style="width: 250px;">
                <b>Rank:</b> {i+1}<br>
                <b>Tile ID:</b> {result.tile_id.split("/")[-1]}<br>
                <b>Distance:</b> {result.distance:.4f}<br>
                <b>Similarity:</b> {(1-result.distance):.4f}<br>
                <b>Center:</b> {lat:.6f}, {lon:.6f}<br>
                <b>Bbox Size:</b> {bbox_width:.6f} x {bbox_height:.6f}<br>
                <b>Acquisition:</b> {result.acquisition_time}
            </div>
            """

            # Color based on rank
            color = 'red' if i == 0 else 'blue' if i < 3 else 'green'

            # Add marker
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Rank {i+1}: {result.tile_id.split('/')[-1]}",
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(m)

            # Add bounding box rectangle with thick, visible styling
            folium.Rectangle(
                bounds=[[result.bbox_south, result.bbox_west],
                       [result.bbox_north, result.bbox_east]],
                color=color,
                weight=5,
                opacity=1.0,
                fill=True,
                fillColor=color,
                fillOpacity=0.5,
                popup=folium.Popup(
                    f"<b>Tile {i+1}</b><br>Size: {bbox_width:.6f}° x {bbox_height:.6f}°",
                    max_width=200
                )
            ).add_to(m)

            # Add a larger circle marker at the center for better visibility
            folium.CircleMarker(
                [lat, lon],
                radius=8,
                popup=f"Center of Tile {i+1}",
                color=color,
                weight=3,
                opacity=1.0,
                fillOpacity=0.8
            ).add_to(m)

        except Exception as e:
            st.warning(f"Error adding result {i+1} to map: {e}")
            continue

    return m


def main():
    # Header
    st.markdown("<h1 class='main-header'>🛰️ NAIP Satellite Image Search with Natural Language</h1>", unsafe_allow_html=True)
    st.markdown("Search through NAIP satellite imagery using natural language queries powered by vector embeddings.")

    # Load model (cached)
    model, tokenizer, device = load_model()

    if model is None:
        st.error("Model failed to load. Please check the logs above for details.")
        st.info("Try refreshing the page to retry model loading.")
        st.stop()

    # Search query in main area
    query_text = st.text_input(
        "Search Query",
        value=st.session_state.query_text,
        placeholder="Describe what you want to find in satellite images",
        help="Describe what you want to find in the satellite images"
    )

    # Example query chips
    EXAMPLE_QUERIES = [
        "basketball court", "swimming pools", "airplanes",
        "construction sites", "roundabouts", "airport strips",
        "ships and boats", "football field",
    ]
    chip_cols = st.columns(len(EXAMPLE_QUERIES))
    for i, example in enumerate(EXAMPLE_QUERIES):
        if chip_cols[i].button(example, key=f"chip_{i}"):
            st.session_state.query_text = example
            st.rerun()

    # Sidebar for parameters
    st.sidebar.header("Search Parameters")

    top_k = st.sidebar.slider(
        "Number of Results",
        min_value=1, max_value=50, value=15,
        help="How many similar images to return"
    )

    distance_threshold = st.sidebar.slider(
        "Distance Threshold",
        min_value=0.1, max_value=1.0, value=0.8, step=0.1,
        help="Maximum distance for results (lower = more similar)"
    )

    fraction_lists_to_search = st.sidebar.select_slider(
        "Search Precision",
        options=[0.001, 0.01, 0.05, 0.1],
        value=0.01,
        format_func=lambda x: f"{x:.1%} ({'Fast' if x <= 0.01 else 'Balanced' if x <= 0.05 else 'Thorough'})",
        help="Trade-off between speed and accuracy"
    )

    # Search button
    search_clicked = st.button("Search Images", type="primary")

    # Perform search
    if search_clicked and query_text:
        start_time = time.time()

        results = text_search_bq(
            query_text, model, tokenizer, device,
            PROJECT_NAME, DATASET_ID, TABLE_ID,
            top_k, distance_threshold, fraction_lists_to_search
        )

        search_time = time.time() - start_time
        st.session_state.search_results = results

        if results:
            st.success(f"Found {len(results)} results in {search_time:.2f} seconds!")
        else:
            st.warning("No results found. Try adjusting the distance threshold or search query.")

    # Display results
    if st.session_state.search_results:
        results = st.session_state.search_results

        if results:
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["🖼️ Image Gallery", "🗺️ Map View", "📋 Details"])

            with tab1:
                st.subheader(f"Search Results for: '{query_text}'")

                # Display images in a grid
                cols_per_row = 3
                for i in range(0, len(results), cols_per_row):
                    cols = st.columns(cols_per_row)

                    for j, col in enumerate(cols):
                        if i + j < len(results):
                            result = results[i + j]

                            with col:
                                # Load and display image
                                img = load_image_from_gcs(result.tile_id, BUCKET_NAME)

                                if img:
                                    st.image(img, use_container_width=True)

                                    # Image info
                                    similarity = (1 - result.distance) * 100
                                    st.caption(f"""
                                    **Rank {i+j+1}** | Similarity: {similarity:.1f}%
                                    📍 Lat: {(result.bbox_south + result.bbox_north)/2:.4f}
                                    📍 Lon: {(result.bbox_west + result.bbox_east)/2:.4f}
                                    📅 {result.acquisition_time}
                                    """)
                                else:
                                    st.error(f"Could not load image {result.tile_id}")

            with tab2:
                st.subheader("📍 Result Locations")

                # Create and display map
                map_obj = create_results_map(results)

                if map_obj:
                    st_folium(map_obj, width=1000, height=700, key="main_map")
                else:
                    st.error("Could not create map")

            with tab3:
                st.subheader("📊 Detailed Results")

                # Create DataFrame for detailed view
                data = []
                for i, result in enumerate(results):
                    data.append({
                        "Rank": i + 1,
                        "Tile ID": result.tile_id.split('/')[-1],
                        "Distance": f"{result.distance:.4f}",
                        "Similarity (%)": f"{(1-result.distance)*100:.1f}%",
                        "Center Lat": f"{(result.bbox_south + result.bbox_north)/2:.6f}",
                        "Center Lon": f"{(result.bbox_west + result.bbox_east)/2:.6f}",
                        "Acquisition Time": str(result.acquisition_time),
                        "Bbox West": f"{result.bbox_west:.6f}",
                        "Bbox South": f"{result.bbox_south:.6f}",
                        "Bbox East": f"{result.bbox_east:.6f}",
                        "Bbox North": f"{result.bbox_north:.6f}",
                    })

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"naip_search_results_{query_text}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This app searches through NAIP satellite imagery using:
    - 🤖 Vector embeddings for semantic search
    - 📊 BigQuery vector search for scalability
    - 🗺️ Interactive maps for geospatial visualization

    **Tips:**
    - Try queries like "airports", "residential areas", "forests"
    - Adjust the distance threshold if you get too few/many results
    - Use the map view to understand geographic distribution
    """)


if __name__ == "__main__":
    main()
