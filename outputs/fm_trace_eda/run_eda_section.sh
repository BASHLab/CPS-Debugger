#!/bin/bash
#SBATCH --partition=quick
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=architecture_eda/slurm_%x_%j.log

# Usage: sbatch --job-name=eda_s1_2_6_10 --mem=8G run_eda_section.sh 1,2,6,10

SECTIONS=$1
cd /home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda

echo "=== Architecture EDA sections: $SECTIONS ==="
echo "Node: $(hostname), Date: $(date)"

python3 architecture_eda.py --sections "$SECTIONS"

echo "=== Done: $(date) ==="
