#!/bin/bash
# Parallel ranged download of ROAD from Zenodo (single-stream is ~85 KB/s).
URL="https://zenodo.org/records/10462796/files/road.zip?download=1"
SIZE=556718276; N=16; cd "$(dirname "$0")"; mkdir -p parts; rm -f road.zip
CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$(( s+CH-1 )); [ $e -ge $SIZE ] && e=$((SIZE-1)); want=$((e-s+1))
  ( f=parts/p$(printf %02d $i)
    while :; do
      have=$( [ -f $f ] && stat -c %s $f || echo 0 )
      [ "$have" -ge "$want" ] && break
      curl -s -L --retry 5 -m 3600 -r $((s+have))-$e "$URL" >> $f
    done ) &
done
wait
cat parts/p* > road.zip && rm -rf parts
ls -la road.zip; md5sum road.zip; sha256sum road.zip | tee road.zip.sha256
unzip -q road.zip -d road/ && echo EXTRACTED
