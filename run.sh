#!/bin/bash
# ============================================================================
# HDM Optimization Pipeline - Simplified Runner for Chile/KFC Backtesting
# ============================================================================
#
# Usage:
#   ./run.sh <FRANCHISE> <GRADE> <START_DATE> <END_DATE> [DAYS_OF_WEEK]
#
# Examples:
#   ./run.sh KFC AAA 2026-02-23 2026-03-01
#   ./run.sh "Burger King" AA 2026-02-16 2026-03-01
#   ./run.sh NIU AA 2026-01-09 2026-03-18 "1,6,7"
#   ./run.sh NIU AA 2026-01-09 2026-03-18 "lun,mar,mie"
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

if [ $# -lt 4 ] || [ $# -gt 5 ]; then
  echo "Usage: $0 <FRANCHISE> <GRADE> <START_DATE> <END_DATE> [DAYS_OF_WEEK]"
    echo ""
    echo "Example:"
    echo "  $0 KFC AAA 2026-02-23 2026-03-01"
  echo "  $0 NIU AA 2026-01-09 2026-03-18 \"1,6,7\""
    echo ""
    echo "Parameters:"
    echo "  FRANCHISE:   Company name (e.g., KFC, Burger King, etc.)"
    echo "  GRADE:       Vendor grade (AAA, AA, or A)"
    echo "  START_DATE:  Start date in YYYY-MM-DD format"
    echo "  END_DATE:    End date in YYYY-MM-DD format"
  echo "  DAYS_OF_WEEK (optional): all | 1-7 | mon,tue,... | lun,mar,..."
  echo "                           BigQuery convention: 1=Sunday ... 7=Saturday"
    exit 1
fi

FRANCHISE="$1"
GRADE="$2"
START_DATE="$3"
END_DATE="$4"
DAYS_OF_WEEK="${5:-all}"

# No profile defaults here: use config.py defaults unless env overrides are set.

echo "=========================================="
echo "HDM Optimization Pipeline - Franchise Mode"
echo "=========================================="
echo "Franchise:  $FRANCHISE"
echo "Grade:      $GRADE"
echo "Date Range: $START_DATE to $END_DATE"
echo "Days:       $DAYS_OF_WEEK"
echo "Country:    Chile (ID: 2)"
echo "Entity:     PY_CL"
echo "GCP Project: ${GOOGLE_CLOUD_PROJECT:-$GCP_PROJECT_ID}"
if [ -n "${N_SIMULATIONS}" ]; then
  echo "N_SIMULATIONS (env override): ${N_SIMULATIONS}"
else
  echo "N_SIMULATIONS: config.py default"
fi
if [ -n "${N_OPTIMIZATION_CALLS}" ]; then
  echo "N_OPTIMIZATION_CALLS (env override): ${N_OPTIMIZATION_CALLS}"
else
  echo "N_OPTIMIZATION_CALLS: config.py default"
fi
if [ -n "${HDM_SIM_N_JOBS}" ]; then
  echo "HDM_SIM_N_JOBS (env override): ${HDM_SIM_N_JOBS}"
else
  echo "HDM_SIM_N_JOBS: config.py/runtime default"
fi
if [ -n "${BQ_TIMEOUT_SECONDS}" ]; then
  echo "BQ_TIMEOUT_SECONDS (env override): ${BQ_TIMEOUT_SECONDS}"
else
  echo "BQ_TIMEOUT_SECONDS: config.py default"
fi
echo "=========================================="
echo ""

python main.py \
  --franchise "$FRANCHISE" \
  --grade "$GRADE" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --days-of-week "$DAYS_OF_WEEK"

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
