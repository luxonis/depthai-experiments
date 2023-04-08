#!/usr/bin/env python3

import os
import sys
import time
import glob
import re
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse


SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_latest_dir(parent_dir):
    max_id = 0
    latest_dir = None

    for entry in os.scandir(parent_dir):
        if entry.is_dir():
            try:
                entry_id = int(entry.name.split('-')[0])
            except ValueError:
                continue

            if entry_id > max_id:
                max_id = entry_id
                latest_dir = entry.path

    return latest_dir


def main():
    parser = argparse.ArgumentParser(description="Script to process camera data.")
    parser.add_argument("calib_dir", help="Path to the calibration directory")
    parser.add_argument("--continue", dest="continue_from_last", action="store_true", help="Continue from the last checkpoint")
    args = parser.parse_args()

    CALIB_DIR = args.calib_dir
    checkpoint_file = '.checkpoint.txt'
    errors_file = '.errors.txt'
    with open(errors_file, 'w') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def read_checkpoint():
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                return f.read().strip()
        return ''

    def save_checkpoint(cam, path, calib_file, position):
        with open(checkpoint_file, 'w') as f:
            f.write(f"{cam},{path},{calib_file},{position}")

    def save_errors(cam, path, calib_file, position):
        with open(errors_file, 'a') as f:
            f.write(f"{cam},{path},{calib_file},{position}")

    cams = [f"camera_{i}" for i in [10, 11]]
    positions = ["center", "left", "right", "top", "bottom"]

    last_checkpoint = read_checkpoint()
    if args.continue_from_last and last_checkpoint:
        last_cam, last_path, last_calib_file, last_position = last_checkpoint.split(',')

    start_from_last = args.continue_from_last and last_checkpoint != ''
    total_iterations = 0
    for cam in cams:
        total_iterations += len(glob.glob(os.path.join(CALIB_DIR, cam, '*'))) * len(glob.glob(os.path.join(CALIB_DIR, cam, 'calibrations', '*'))) * len(positions)
    progress_bar = tqdm(total=total_iterations, desc="Processing", ncols=100)

    try:
        for cam in cams:
            for path in glob.glob(os.path.join(CALIB_DIR, cam, '*')):
                path_basename = os.path.basename(path)

                if not os.path.isdir(path) or not re.match(r'^\d+(\.\d+)?m$', path_basename):
                    print(f"Skipping {path_basename}")
                    continue

                for calib_file in glob.glob(os.path.join(CALIB_DIR, cam, 'calibrations', '*')):
                    for position in positions:
                        if start_from_last:
                            if (cam, path, calib_file, position) != (last_cam, last_path, last_calib_file, last_position):
                                continue
                            else:
                                start_from_last = False

                        print(f"Running in {cam} {path} m {position}")
                        d_dir = os.path.join(path, position)

                        if not os.path.isdir(d_dir):
                            print(f"Directory {d_dir} does not exist")
                            continue

                        latest_dir = get_latest_dir(d_dir)
                        if latest_dir is None:
                            print("No latest dir found")
                            continue

                        if not os.path.isdir(latest_dir):
                            print(f"Directory {latest_dir} does not exist")
                            continue

                        print(f"Base dir is {latest_dir}")

                        calib_name = os.path.basename(calib_file)
                        calib_name_no_ext = os.path.splitext(calib_name)[0]
                        progress_bar.set_description(f"Processing: {cam} {os.path.basename(path)} {os.path.basename(calib_file)} {position}")
                        progress_bar.refresh()

                        for res_type, res_arg in [("fullRes", "-fullResolution"), ("resizedRes", "")]:
                            out_dir = os.path.join(latest_dir, res_type, calib_name_no_ext)
                            if Path(out_dir).exists():
                                print("Skipping existing directory")
                                continue
                            os.makedirs(out_dir, exist_ok=True)

                            print(f"Running {res_type}")

                            cmd = f"python {SOURCE_DIR}/stereo_both.py -vid {res_arg} -saveFiles -numLastFrames 10 -imageCrop {position} -left {latest_dir}/camb,c.avi -right {latest_dir}/camc,c.avi -bottom {latest_dir}/camd,c.avi -calib {calib_file} -rect -outDir {out_dir}"
                            ret = subprocess.run(cmd, shell=True, capture_output=True)
                            if ret.returncode != 0:
                                print(f"Error running {cmd}")
                                save_errors(cam, path, calib_file, position)
                                print(ret.stderr)
                                print(ret.stdout)
                            print(cmd)
                            time.sleep(10)
                            progress_bar.update(1)
                        save_checkpoint(cam, path, calib_file, position)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")
        save_checkpoint(cam, path, calib_file, position)
        print("Progress saved. Exiting.")
        exit(1)
    print("Script completed successfully.")
    progress_bar.close()


if __name__ == "__main__":
    main()
