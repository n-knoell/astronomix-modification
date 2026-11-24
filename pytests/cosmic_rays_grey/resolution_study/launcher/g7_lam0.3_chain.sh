#!/bin/bash
# 2026-11-18: G7 at 384^3 + lambda 0.3 at 384^3 for G4-G6, then galleries.
# 1. gpu_lanes.py --max 2 on queue_g7_lam0.3_20261118.txt (7 runs, free GPUs only, kept between jobs);
# 2. gpu_lanes.py --max 1 on the post queue: R_g7_lam0.3_gpu (gallery_d_extract --n 384, gallery_g7 extract --n 384);
# 3. back up pics/G3-G7 to RAW/pics_pre_g7_lam0.3_20261118/, then CPU: gallery_d_plot.py (G3-G6),
#    gallery_g7.py plot --n 384 (-> pics/G7_cwb_diffusion_gallery_N384.png).
# Usage: setsid nohup bash launcher/g7_lam0.3_chain.sh > RAW/logs/g7_lam0.3_chain.log 2>&1 &
RAW=/export/scratch/nknoell/resolution_study
HERE=/export/home/nknoell/astronomix-modification/pytests/cosmic_rays_grey/resolution_study
source ~/venv_grav_source/bin/activate
export PYTHONUNBUFFERED=1
cd "$HERE"
echo "CHAIN START $(date '+%F %T') host $(hostname) pid $$"
python launcher/gpu_lanes.py "$RAW/jobs/queue_g7_lam0.3_20261118.txt" --max 2
for t in $(sed 's/#.*//' "$RAW/jobs/queue_g7_lam0.3_20261118.txt"); do
  echo "  $t rc=$(cat "$RAW/logs/done/$t" 2>/dev/null || echo missing): $(grep -E '^saved|FAILED|non-finite' "$RAW/logs/$t.log" 2>/dev/null | tail -1 | cut -c1-150)"
done
echo "RUNS DONE $(date '+%F %T')"
python launcher/gpu_lanes.py "$RAW/jobs/queue_g7_lam0.3_20261118_post.txt" --max 1
gpu_rc=$(cat "$RAW/logs/done/R_g7_lam0.3_gpu" 2>/dev/null || echo missing)
echo "GPU PART DONE $(date '+%F %T') rc=$gpu_rc (log logs/R_g7_lam0.3_gpu.log)"
mkdir -p "$RAW/pics_pre_g7_lam0.3_20261118"; cp -a pics/G[3-7]_*.png "$RAW/pics_pre_g7_lam0.3_20261118/"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
failed=()
for cmd in "gallery_d_plot.py" "gallery_g7.py plot --n 384"; do
  echo "=== python $cmd $(date '+%F %T')"
  python $cmd || { echo "FAILED rc=$?: $cmd"; failed+=("$cmd"); }
done
printf "gpu_rc=%s cpu_failed=%s\n" "$gpu_rc" "${failed[*]:-none}" > "$RAW/logs/done/R_g7_lam0.3_all"
echo "CHAIN DONE $(date '+%F %T') failed: ${failed[*]:-none}"
