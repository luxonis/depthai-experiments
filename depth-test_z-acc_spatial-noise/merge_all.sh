#!/usr/bin/env bash

SOURCE_DIR=$(dirname "$0")
CALIB_DIR=$1

echo "Calib dir is $CALIB_DIR"
TARGET_FILE=$CALIB_DIR/merged_depth_results.csv

for cam in {1..3}; do
    for dist in 1 2 4; do
        for position in left right center; do
            echo "Running in $cam $dist m $position"
            dDir=$CALIB_DIR/cam_$cam/${dist}m/$position
            echo "Base dir is $dDir"
            python $SOURCE_DIR/merge_results.py --type vertical --camId $cam --side $position --pathTo $TARGET_FILE --pathFrom $dDir/results_ver.txt
            python $SOURCE_DIR/merge_results.py --type horizontal --camId $cam --side $position --pathTo $TARGET_FILE --pathFrom $dDir/results_hor.txt
        done
    done
done
