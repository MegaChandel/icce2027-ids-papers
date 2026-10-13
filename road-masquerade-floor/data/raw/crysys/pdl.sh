#!/bin/bash
# Parallel ranged download of the CrySyS CAN dataset (figshare -> S3), CC0.
cd "$(dirname "$0")"
URL=$(curl -s -o /dev/null -w "%{redirect_url}" https://ndownloader.figshare.com/files/41453310)
SIZE=2848034718; N=16; mkdir -p parts
CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SIZE ] && e=$((SIZE-1)); want=$((e-s+1))
  ( f=parts/p$(printf %02d $i)
    while :; do
      have=$( [ -f $f ] && stat -c %s $f || echo 0 ); [ "$have" -ge "$want" ] && break
      curl -s -L --retry 5 -m 3600 -r $((s+have))-$e "$URL" >> $f
    done ) &
done
wait
cat parts/p* > crysys.zip && rm -rf parts
md5sum crysys.zip   # expect 789735316f77ebee0f9f52116fad3cf7
unzip -q crysys.zip -d . && echo EXTRACTED
