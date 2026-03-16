#!/bin/bash
# ============================================================================
# HDM Optimization Pipeline - Production Profile Runner
# ============================================================================
# Usage:
#   bash ./run_prod.sh <FRANCHISE> <GRADE> <START_DATE> <END_DATE>
#
# Full production profile.
# Profile:
#   - N_SIMULATIONS=2000
#   - N_OPTIMIZATION_CALLS=80
#   - HDM_SIM_N_JOBS=1 (safe default for JupyterHub)
# ============================================================================

set -e

export N_SIMULATIONS="${N_SIMULATIONS:-2000}"
export N_OPTIMIZATION_CALLS="${N_OPTIMIZATION_CALLS:-80}"
export HDM_SIM_N_JOBS="${HDM_SIM_N_JOBS:-1}"

bash ./run.sh "$@"
