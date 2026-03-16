#!/bin/bash
# ============================================================================
# HDM Optimization Pipeline - Simplified Runner for Chile/KFC Backtesting
# ============================================================================
#
# Usage:
#   ./run.sh <FRANCHISE> <GRADE> <START_DATE> <END_DATE>
#
# Examples:
#   ./run.sh KFC AAA 2026-02-23 2026-03-01
#   ./run.sh "Burger King" AA 2026-02-16 2026-03-01
#
# Notes:
#   - Country ID is fixed to 2 (Chile)
#   - Entity ID is fixed to PY_CL
#   - Data source is BigQuery
#   - Dates must be in YYYY-MM-DD format
#   - Requires GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT environment variable
#
# ============================================================================

set -e

# Validate that required env vars are set
if [ -z "$GOOGLE_CLOUD_PROJECT" ] && [ -z "$GCP_PROJECT_ID" ]; then
    echo "ERROR: Missing GCP project configuration!"
    echo ""
    echo "Please set the following environment variables:"
    echo "  export GOOGLE_CLOUD_PROJECT=\"peya-chile\""
    echo "  export GCP_PROJECT_ID=\"peya-chile\""
    echo "  export BQ_LOCATION=\"US\""
    echo ""
    echo "Then run this script again:"
    echo "  $0 $@"
    exit 1
fi

if [ $# -ne 4 ]; then
    echo "Usage: $0 <FRANCHISE> <GRADE> <START_DATE> <END_DATE>"
    echo ""
    echo "Example:"
    echo "  $0 KFC AAA 2026-02-23 2026-03-01"
    echo ""
    echo "Parameters:"
    echo "  FRANCHISE:   Company name (e.g., KFC, Burger King, etc.)"
    echo "  GRADE:       Vendor grade (AAA, AA, or A)"
    echo "  START_DATE:  Start date in YYYY-MM-DD format"
    echo "  END_DATE:    End date in YYYY-MM-DD format"
    exit 1
fi

FRANCHISE="$1"
GRADE="$2"
START_DATE="$3"
END_DATE="$4"

# Smoke-test defaults (can be overridden from shell before execution)
export N_SIMULATIONS="${N_SIMULATIONS:-10}"
export N_OPTIMIZATION_CALLS="${N_OPTIMIZATION_CALLS:-5}"
export HDM_SIM_N_JOBS="${HDM_SIM_N_JOBS:-1}"

echo "=========================================="
echo "HDM Optimization Pipeline - Franchise Mode"
echo "=========================================="
echo "Franchise:  $FRANCHISE"
echo "Grade:      $GRADE"
echo "Date Range: $START_DATE to $END_DATE"
echo "Country:    Chile (ID: 2)"
echo "Entity:     PY_CL"
echo "GCP Project: ${GOOGLE_CLOUD_PROJECT:-$GCP_PROJECT_ID}"
echo "N_SIMULATIONS: ${N_SIMULATIONS}"
echo "N_OPTIMIZATION_CALLS: ${N_OPTIMIZATION_CALLS}"
echo "HDM_SIM_N_JOBS: ${HDM_SIM_N_JOBS}"
echo "=========================================="
echo ""

python main.py \
  --franchise "$FRANCHISE" \
  --grade "$GRADE" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE"

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
