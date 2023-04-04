#!/usr/bin/env bash

SOURCE_DIR=$(dirname "$0")
CALIB_DIR=$1

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

echo "Calib dir is $CALIB_DIR"
TARGET_FILE=$CALIB_DIR/merged_depth_results.csv


for cam in 5; do
    for path in $CALIB_DIR/camera_$cam/*; do
        last_directory=$(basename "$path")
        gt=$(echo "$last_directory" | grep -o -E '[+-]?[0-9]+([.][0-9]+)?')
        echo "Last directory is $last_directory"
        echo "GT is $gt"
        #for position in left right top down; do
        for position in center left right top bottom; do
            for resolution in fullRes resizedRes; do
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
                outDir=$dDir/$resolution
                python $SOURCE_DIR/merge_results.py --resolution $resolution --type vertical --camId $cam --side $position --pathTo $TARGET_FILE --pathFrom $outDir/results_ver.txt
                python $SOURCE_DIR/merge_results.py --resolution $resolution --type horizontal --camId $cam --side $position --pathTo $TARGET_FILE --pathFrom $outDir/results_hor.txt
            done
        done
    done
done
