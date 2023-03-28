#!/usr/bin/env bash

SOURCE_DIR=$(dirname "$0")
CALIB_DIR=$1

echo "Calib dir is $CALIB_DIR"

for cam in cam_{1..3}; do
    for dist in 1 2 4; do
        for position in left right center; do
            echo "Running in $cam $dist m $position"
            dDir=$CALIB_DIR/$cam/${dist}m/$position
            echo "Base dir is $dDir"
            echo "Vertical"
            python $SOURCE_DIR/main.py -calib $CALIB_DIR/$cam/vermeer.json -depth $dDir/outDepthVer.npy -rectified $dDir/outLeftRectVer.npy -gt $dist -out_results_f $dDir/results_ver.txt
            echo "Horizontal"
            python $SOURCE_DIR/main.py -calib $CALIB_DIR/$cam/vermeer.json -depth $dDir/outDepthHor.npy -rectified $dDir/outLeftRectHor.npy -gt $dist -out_results_f $dDir/results_hor.txt
        done
    done
done
