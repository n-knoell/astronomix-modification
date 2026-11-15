#!/bin/bash
# Re-analysis after the H200 extension (EXTENSION_KICKSTART.md "Analysis after the runs").
# 1. GPU part: gallery_extract.py + gallery_d_extract.py --n 320 384 as job R_gallery_extract,
#    dispatched by gpu_lanes.py (--max 1): waits until a GPU has no compute process at all.
# 2. CPU part (after the GPU is released; JAX forced to CPU, no GPU context): s1-s5, c_analyse,
#    d_analyse, d_plots, gallery_plot, gallery_d_plot -> figures/tables in pics/.
# Usage: setsid nohup bash launcher/reanalysis_watch.sh > RAW/logs/reanalysis.log 2>&1 &
RAW=/export/scratch/nknoell/resolution_study
HERE=/export/home/nknoell/astronomix-modification/pytests/cosmic_rays_grey/resolution_study
source ~/venv_grav_source/bin/activate
export PYTHONUNBUFFERED=1
cd "$HERE"
echo "REANALYSIS START $(date '+%F %T') host $(hostname) pid $$"

python launcher/gpu_lanes.py "$RAW/jobs/queue_reanalysis.txt" --max 1
gpu_rc=$(cat "$RAW/logs/done/R_gallery_extract" 2>/dev/null || echo missing)
echo "GPU PART DONE $(date '+%F %T') rc=$gpu_rc (log logs/R_gallery_extract.log)"

export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
failed=()
for cmd in "s1_analyse.py" "s2_analyse.py" "s3_analyse.py" "s4_analyse.py" "s5_summary.py" \
           "c_analyse.py" "d_analyse.py" "d_plots.py" "gallery_plot.py" "gallery_d_plot.py"; do
  echo "=== python $cmd $(date '+%F %T')"
  python $cmd || { echo "FAILED rc=$?: $cmd"; failed+=("$cmd"); }
done
echo "CPU PART DONE $(date '+%F %T') failed: ${failed[*]:-none}"
printf "gpu_rc=%s cpu_failed=%s\n" "$gpu_rc" "${failed[*]:-none}" > "$RAW/logs/done/R_reanalysis_all"
echo "REANALYSIS DONE $(date '+%F %T')"
