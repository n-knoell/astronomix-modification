#!/bin/bash
# usage: launch.sh <tag> <GPU-uuid>: runs jobs/<tag>.sh pinned to that GPU, log logs/<tag>.log.
# Refuses if the GPU is busy right now (GPU 0: two idle jupyter kernels of another user,
# PIDs 2989099 / 3160311, ~234 MiB each, tolerated if utilization is 0).
R=/export/scratch/nknoell/resolution_study; tag=$1; u=$2
[ -x $R/jobs/$tag.sh ] || { echo "no executable $R/jobs/$tag.sh"; exit 2; }
read -r mem util <<< "$(nvidia-smi --id=$u --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')"
apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep "$u" | grep -vc "^2989099,\|^3160311,")
if [ "$apps" != "0" ] || [ "$mem" -gt 600 ] || [ "$util" -gt 5 ]; then echo "REFUSE $tag on $u: apps=$apps mem=${mem}MiB util=$util"; exit 1; fi
CUDA_VISIBLE_DEVICES=$u nohup $R/jobs/$tag.sh > $R/logs/$tag.log 2>&1 &
echo "$tag pid=$! uuid=$u $(date '+%F %T')" | tee -a $R/logs/launches.txt
