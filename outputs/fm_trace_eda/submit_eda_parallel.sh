#!/bin/bash
# Submit architecture EDA sections as parallel SLURM jobs.
# Groups are independent — no cross-dependencies.

cd /home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda
mkdir -p architecture_eda

# Group A: fast sections (datalayer + layout.json, ~1 min)
JA=$(sbatch --job-name=eda_fast --mem=4G \
     run_eda_section.sh "1,2,6,10" | awk '{print $4}')
echo "Group A (sections 1,2,6,10): $JA"

# Group B: variable edges (3 sessions, ~5 min)
JB=$(sbatch --job-name=eda_var_edges --mem=16G \
     run_eda_section.sh "3" | awk '{print $4}')
echo "Group B (section 3): $JB"

# Group C: autocorrelation (1 session contiguous, ~3 min)
JC=$(sbatch --job-name=eda_autocorr --mem=16G \
     run_eda_section.sh "4" | awk '{print $4}')
echo "Group C (section 4): $JC"

# Group D: sensor-trace correlation + FSM (3 sessions merged, ~10 min)
JD=$(sbatch --job-name=eda_corr_fsm --mem=16G \
     run_eda_section.sh "5,7" | awk '{print $4}')
echo "Group D (sections 5,7): $JD"

# Group E: cross-session consistency (all sessions skiprows, ~15 min)
JE=$(sbatch --job-name=eda_cross_sess --mem=16G \
     run_eda_section.sh "8" | awk '{print $4}')
echo "Group E (section 8): $JE"

# Group F: sequence lengths (5 sessions, ~8 min)
JF=$(sbatch --job-name=eda_seq_len --mem=16G \
     run_eda_section.sh "9" | awk '{print $4}')
echo "Group F (section 9): $JF"

# Collect job: assemble report + figures after all complete
JCOL=$(sbatch --job-name=eda_collect --mem=4G \
       --dependency=afterok:${JA}:${JB}:${JC}:${JD}:${JE}:${JF} \
       run_eda_section.sh "collect" | awk '{print $4}')
echo "Collect (after all): $JCOL"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Collect job $JCOL will generate final report after all sections complete."
