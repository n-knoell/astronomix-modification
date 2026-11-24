#!/bin/bash
# Re-analysis chained after the H200 reruns of RERUN_LIST.md (2026-11-17).
# 0. wait until the rerun dispatcher (gpu_lanes.py on queue_rerun20261116_h200.txt) has exited;
#    abort if a core rerun failed but left an output file (old/partial state -> mixed analysis);
# 1. GPU (gpu_lanes.py --max 1, job R_rerun20261116_gpu): gallery_extract, gallery_d_extract
#    --n 320 384, c2_restrict, x2_cwb, c_expect;
# 2. back up pics/ to RAW/pics_pre_rerun20261116/, then CPU (JAX on CPU): s1-s5, c_analyse,
#    d_analyse, d_plots, gallery_plot, gallery_d_plot -> pics/.
# Usage: setsid nohup bash launcher/rerun_reanalysis_chain.sh <dispatcher pid> > RAW/logs/rerun20261116_reanalysis.log 2>&1 &
RAW=/export/scratch/nknoell/resolution_study
HERE=/export/home/nknoell/astronomix-modification/pytests/cosmic_rays_grey/resolution_study
QUEUE=$RAW/jobs/queue_rerun20261116_h200.txt
DPID=$1
source ~/venv_grav_source/bin/activate
export PYTHONUNBUFFERED=1
cd "$HERE"
echo "CHAIN START $(date '+%F %T') host $(hostname) pid $$, waiting for dispatcher pid $DPID"
while kill -0 "$DPID" 2>/dev/null; do sleep 60; done
echo "RERUNS FINISHED $(date '+%F %T')"

# output file of each core rerun (DB12 probes excluded: they had no output before)
out() {
  case $1 in
    H_ss_z0.1_lam*_N*) l=${1#H_ss_z0.1_lam}; echo "sedov/ss/N${l#*_N}/z0.1_lam${l%_N*}.npz" ;;
    H_cwb_z0.1_N256)   echo "cwb/N256/z0.1.npz" ;;
    H_cwb_control_N256) echo "cwb/N256/control.npz" ;;
    H_w_*_N*)          w=${1#H_w_}; echo "wind/N${w#*_N}/${w%_N*}.npz" ;;
  esac
}
bad=()
for t in $(sed 's/#.*//' "$QUEUE"); do
  rc=$(cat "$RAW/logs/done/$t" 2>/dev/null || echo missing)
  f=$(out "$t"); [ -z "$f" ] && { echo "  $t rc=$rc (DB12 probe)"; continue; }
  if [ "$rc" = 0 ] && [ "$RAW/$f" -nt "$QUEUE" ]; then echo "  $t rc=0 $f new"
  elif [ "$rc" != 0 ] && [ ! -e "$RAW/$f" ]; then echo "  $t rc=$rc, no $f (no old file either; analysis goes on without it)"
  else echo "  $t rc=$rc, $f stale or partial"; bad+=("$t"); fi
done
if [ ${#bad[@]} -gt 0 ]; then
  echo "ABORT $(date '+%F %T'): re-analysis would mix old and new states: ${bad[*]}"
  printf "aborted: %s\n" "${bad[*]}" > "$RAW/logs/done/R_rerun20261116_reanalysis_all"; exit 1
fi

python launcher/gpu_lanes.py "$RAW/jobs/queue_rerun20261116_reanalysis.txt" --max 1
gpu_rc=$(cat "$RAW/logs/done/R_rerun20261116_gpu" 2>/dev/null || echo missing)
echo "GPU PART DONE $(date '+%F %T') rc=$gpu_rc (log logs/R_rerun20261116_gpu.log)"

rm -rf "$RAW/pics_pre_rerun20261116"; cp -a pics "$RAW/pics_pre_rerun20261116"
echo "old figures backed up to $RAW/pics_pre_rerun20261116"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
failed=()
for cmd in "s1_analyse.py" "s2_analyse.py" "s3_analyse.py" "s4_analyse.py" "s5_summary.py" \
           "c_analyse.py" "d_analyse.py" "d_plots.py" "gallery_plot.py" "gallery_d_plot.py"; do
  echo "=== python $cmd $(date '+%F %T')"
  python $cmd || { echo "FAILED rc=$?: $cmd"; failed+=("$cmd"); }
done
echo "CPU PART DONE $(date '+%F %T') failed: ${failed[*]:-none}"
printf "gpu_rc=%s cpu_failed=%s\n" "$gpu_rc" "${failed[*]:-none}" > "$RAW/logs/done/R_rerun20261116_reanalysis_all"
echo "REANALYSIS DONE $(date '+%F %T')"
