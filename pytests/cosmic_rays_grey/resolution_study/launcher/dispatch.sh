#!/bin/bash
# usage: GPUS="GPU-a GPU-b ..." dispatch.sh <tag> ...: launches the queued jobs in order, one per
# idle GPU of the list, skipping GPUs held by my still-running job shells; verifies each pin.
R=/export/scratch/nknoell/resolution_study
mine_busy() { [ -f $R/logs/launches.txt ] && while read -r tag pidf uuidf d t; do ps -p ${pidf#pid=} >/dev/null && echo ${uuidf#uuid=}; done < $R/logs/launches.txt; }
idle_gpu() { busy="$(mine_busy)"
  for u in $GPUS; do echo "$busy" | grep -q "$u" && continue
    $R/launch.sh "$1" "$u" >/dev/null 2>&1 && { echo "$u"; return; }; done; }
for tag in "$@"; do
  while true; do u=$(idle_gpu $tag); [ -n "$u" ] && break; sleep 30; done
  echo "launched $tag on $u $(date '+%F %T')"; sleep 90
  echo "VERIFY $($R/verify.sh $tag)"
done
echo "DISPATCH DONE $(date '+%F %T')"
