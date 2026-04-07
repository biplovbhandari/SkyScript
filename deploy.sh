#!/bin/bash
set -e

# Load config from .env if present
if [ -f .env ]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        key="$(echo "$key" | xargs)"
        export "$key=$value"
    done < .env
fi

# Configuration (set via .env or environment)
: "${PROJECT_NAME:?Error: PROJECT_NAME is not set. Add it to .env or export it.}"
: "${BUCKET_NAME:?Error: BUCKET_NAME is not set. Add it to .env or export it.}"
: "${DATASET_ID:?Error: DATASET_ID is not set. Add it to .env or export it.}"
: "${TABLE_ID:?Error: TABLE_ID is not set. Add it to .env or export it.}"

PROJECT_ID="$PROJECT_NAME"
SERVICE_NAME="naip-satellite-search"
REGION="us-central1"
REPO_NAME="skyscript"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}"

usage() {
    echo "Usage: ./deploy.sh [command]"
    echo ""
    echo "Commands:"
    echo "  deploy   Build and deploy to Cloud Run"
    echo "  stop     Delete the Cloud Run service"
    echo "  clean    Delete service, images, and build artifacts"
    echo "  status   Show service status and URL"
    echo "  logs     Tail service logs"
    echo ""
}


do_deploy() {
    echo "Deploying to Cloud Run..."
    echo "  Project: ${PROJECT_ID}"
    echo "  Service: ${SERVICE_NAME}"
    echo "  Region:  ${REGION}"
    echo ""

    # Set project
    gcloud config set project $PROJECT_ID

    # Enable required APIs
    echo "Enabling APIs..."
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable run.googleapis.com
    gcloud services enable artifactregistry.googleapis.com

    # Create Artifact Registry repo if it doesn't exist
    gcloud artifacts repositories describe $REPO_NAME \
        --location=$REGION --format="value(name)" 2>/dev/null || \
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION

    # Build and push image
    echo "Building Docker image..."
    gcloud builds submit \
        --tag $IMAGE_NAME \
        --timeout=3600s \
        --machine-type=e2-highcpu-32 \
        --disk-size=250GB

    # Deploy to Cloud Run
    echo "Deploying to Cloud Run..."
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_NAME \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 32Gi \
        --cpu 8 \
        --timeout 1800 \
        --concurrency 80 \
        --max-instances 3 \
        --min-instances 0 \
        --set-env-vars="PYTHONPATH=/app/src:/app,PROJECT_NAME=${PROJECT_ID},BUCKET_NAME=${BUCKET_NAME},DATASET_ID=${DATASET_ID},TABLE_ID=${TABLE_ID}" \
        --port 8080

    echo ""
    do_status
    echo ""
    echo "Note: First request takes 2-3 minutes (model loading)"
}


do_stop() {
    echo "Deleting Cloud Run service: ${SERVICE_NAME}..."
    gcloud run services delete $SERVICE_NAME \
        --platform managed \
        --region $REGION \
        --quiet
    echo "Service deleted."
}


do_status() {
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --platform managed --region $REGION \
        --format 'value(status.url)' 2>/dev/null) || true

    if [ -z "$SERVICE_URL" ]; then
        echo "Service '${SERVICE_NAME}' not found in ${REGION}."
    else
        echo "Service: ${SERVICE_NAME}"
        echo "URL:     ${SERVICE_URL}"
        gcloud run services describe $SERVICE_NAME \
            --platform managed --region $REGION \
            --format="table(status.conditions[0].type,status.conditions[0].status)"
    fi
}


do_clean() {
    echo "Cleaning up all resources..."

    # Delete Cloud Run service (if exists)
    gcloud run services delete $SERVICE_NAME \
        --platform managed \
        --region $REGION \
        --quiet 2>/dev/null && echo "Service deleted." || echo "Service not found, skipping."

    # Delete all images from Artifact Registry
    echo "Deleting images from Artifact Registry..."
    gcloud artifacts docker images delete "${IMAGE_NAME}" \
        --delete-tags --quiet 2>/dev/null && echo "Images deleted." || echo "No images found, skipping."

    echo "Clean up complete."
    echo "Note: Cloud Build source tarballs in gs://${PROJECT_ID}_cloudbuild are shared across services and left as-is."
}


do_logs() {
    gcloud beta run services logs tail $SERVICE_NAME --region=$REGION
}


# Route command
case "${1:-}" in
    deploy) do_deploy ;;
    stop)   do_stop ;;
    clean)  do_clean ;;
    status) do_status ;;
    logs)   do_logs ;;
    *)      usage ;;
esac
