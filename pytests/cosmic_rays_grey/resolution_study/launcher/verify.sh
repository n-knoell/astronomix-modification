#!/bin/bash
# usage: verify.sh [tag]: for each launch (or only <tag>), the python children and the GPU they run on
R=/export/scratch/nknoell/resolution_study; only=$1
apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null)
while read -r tag pidf uuidf d t; do
  [ -n "$only" ] && [ "$tag" != "$only" ] && continue
  pid=${pidf#pid=}; want=${uuidf#uuid=}
  kids=$(pgrep -P $pid python)
  if [ -z "$kids" ]; then echo "$tag: no python child (shell alive: $(ps -p $pid >/dev/null && echo yes || echo no))"; continue; fi
  for k in $kids; do on=$(echo "$apps" | awk -F', ' -v p=$k '$1==p{print $2}' | tr '\n' ' ')
    st=$([ "$on" = "$want " ] && echo OK || echo CHECK); echo "$tag py=$k want=$want on=[${on}] $st"; done
done < $R/logs/launches.txt
