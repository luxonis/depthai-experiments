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
        last_directory=$(basename "$path")
        gt=$(echo "$last_directory" | grep -o -E '[+-]?[0-9]+([.][0-9]+)?')
        echo "Last directory is $last_directory"
        echo "GT is $gt"
        #for position in left right top down; do
        for position in center; do
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
                echo "Out dir is $outDir"
                echo "Running full resolution"
                echo "Running $position"
                # python $SOURCE_DIR/main.py -calib $dDir/calib.json -depth $outDir/verticalDepth.npy -rectified $outDir/leftRectifiedVertical.npy -gt $gt -out_results_f $outDir/results_ver.txt -roi_file $outDir/roi_vertical.txt
                python $SOURCE_DIR/main.py -mode measure -calib $dDir/calib.json -depth $outDir/verticalDepth.npy -rectified $outDir/leftRectifiedVertical.npy -gt $gt -out_results_f $outDir/results_ver_auto.txt -set_roi_file $outDir/roi_vertical.txt

                echo "Running resized"
                echo "Running $position"
                # python $SOURCE_DIR/main.py -calib $dDir/calib.json -depth $outDir/horizontalDepth.npy -rectified $outDir/leftRectifiedHorizontal.npy -gt $gt -out_results_f $outDir/results_hor.txt -roi_file $outDir/roi_horizontal.txt
                python $SOURCE_DIR/main.py -mode measure -calib $dDir/calib.json -depth $outDir/horizontalDepth.npy -rectified $outDir/leftRectifiedHorizontal.npy -gt $gt -out_results_f $outDir/results_hor_auto.txt -set_roi_file $outDir/roi_horizontal.txt
                # echo "python $SOURCE_DIR/main.py -mode measure -calib $dDir/calib.json -depth $outDir/horizontalDepth.npy -rectified $outDir/leftRectifiedHorizontal.npy -gt $gt -out_results_f $outDir/results_hor_auto.txt -set_roi_file $outDir/roi_horizontal.txt"
            done
        done
    done
done
