#!/bin/bash
# ============================================================================
# Setup Script - Configure environment for HDM Optimization Pipeline
# ============================================================================
#
# This script sets up all required environment variables for JupyterHub.
#
# Usage:
#   source ./setup.sh
#
# ============================================================================

echo "Setting up HDM Optimization Pipeline environment..."

# GCP Configuration
export GOOGLE_CLOUD_PROJECT="peya-chile"
export GCP_PROJECT_ID="peya-chile"
export BQ_LOCATION="US"
export BQ_TIMEOUT_SECONDS=300

echo "✓ GCP_PROJECT_ID = $GCP_PROJECT_ID"
echo "✓ GOOGLE_CLOUD_PROJECT = $GOOGLE_CLOUD_PROJECT"
echo "✓ BQ_LOCATION = $BQ_LOCATION"
echo ""
echo "Environment ready! You can now run:"
echo "  ./run.sh <FRANCHISE> <GRADE> <START_DATE> <END_DATE>"
echo ""
echo "Example:"
echo "  ./run.sh KFC AAA 2026-02-23 2026-03-01"
