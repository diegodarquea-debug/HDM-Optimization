#!/bin/bash
# ============================================================================
# HDM Optimization Pipeline - Realistic Profile Runner
# ============================================================================
# Usage:
#   bash ./run_realistic.sh <FRANCHISE> <GRADE> <START_DATE> <END_DATE>
#
# Recommended for validation with a longer historical window (e.g. 14 days).
# Profile:
#   - N_SIMULATIONS=200
#   - N_OPTIMIZATION_CALLS=20
#   - HDM_SIM_N_JOBS=1
# ============================================================================

set -e

export N_SIMULATIONS="${N_SIMULATIONS:-200}"
export N_OPTIMIZATION_CALLS="${N_OPTIMIZATION_CALLS:-20}"
export HDM_SIM_N_JOBS="${HDM_SIM_N_JOBS:-1}"

bash ./run.sh "$@"
