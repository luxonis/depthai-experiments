#!/usr/bin/env bash

SOURCE_DIR=$(dirname "$0")
CALIB_DIR=$1

echo "Calib dir is $CALIB_DIR"


get_latest_id() {
  parent_dir="$1"
  max_id=0
  dir=""
  for dir in "$parent_dir"/*; do
    if [[ -d $dir ]]; then
        id=$(echo "$(basename $dir)" | cut -d- -f1)
        if (( id > max_id )); then
            max_id=$id
            dir=$dir
        fi
    fi
  done

  echo $dir
}


for cam in camera_{5..6}; do
    for path in $CALIB_DIR/$cam/*; do
        # for position in center; do
        for position in left right top down; do

            echo "Running in $cam $path m $position"
            dDir=$path/$position
            if [ ! -d "$dDir" ]; then
                echo "Directory $dir does not exist"
                continue
            fi

            latestDir=$(get_latest_id $dDir)
            dDir=$latestDir
            if [ ! -d "$dDir" ]; then
                echo "Directory $dir does not exist"
                continue
            fi
            echo "Base dir is $dDir"
            outDir=$dDir/fullRes
            mkdir -p $outDir
            # echo "Base dir is $dDir"
            echo "Running full resolution"
            python $SOURCE_DIR/stereo_both.py -vid -fullResolution -saveFiles -numLastFrames 10 -imageCrop $position -left $dDir/camb,c.avi -right $dDir/camc,c.avi -bottom $dDir/camd,c.avi -calib $dDir/calib.json -rect -outDir $outDir
            sleep 10
            echo "Running resized"
            outDir=$dDir/resizedRes
            mkdir -p $outDir
            python $SOURCE_DIR/stereo_both.py -vid -saveFiles -numLastFrames 10 -imageCrop $position -left $dDir/camb,c.avi -right $dDir/camc,c.avi -bottom $dDir/camd,c.avi -calib $dDir/calib.json -rect -outDir $outDir
            sleep 10

        done
    done
done
