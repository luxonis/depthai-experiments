import os
import sys
import subprocess
from argparse import ArgumentParser
from pathlib import Path


def get_latest_directory(parent_dir):
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
    if latest_dir is None:
        return None
    else:
        return Path(latest_dir)


def merge_results(fromFile, toFile,  camId, result_type, resolution, side, calib):
    if not toFile.exists():
        with open(toFile, "w") as f:
            f.write("cam_id,calib,type,resolution,side,ground_truth,z_measured_mean,z_measured_median,z_accuracy,fillrate,spatial_noise,subpixel_spatial_noise,vertical_tilt,horizontal_tilt")

    if not fromFile.exists():
        print("Skipping file:", fromFile.absolute(), "because it does not exist")
        return

    fromFileSplit = fromFile.read_text().splitlines()[-1].split(",")

    ground_truth = fromFileSplit[1]
    z_measured_mean = fromFileSplit[2]
    z_measured_median = fromFileSplit[3]
    z_accuracy = fromFileSplit[4]
    spatial_noise = fromFileSplit[5]
    subpixel_spatial_noise = fromFileSplit[6]
    fillrate = fromFileSplit[7]
    vertical_tilt = fromFileSplit[8]
    horizontal_tilt = fromFileSplit[9]

    with open(toFile, "a") as f:
        f.write(f"\n{camId},{calib},{result_type},{resolution},{side},{ground_truth},{z_measured_mean},{z_measured_median},{z_accuracy},{fillrate},{spatial_noise},{subpixel_spatial_noise},{vertical_tilt},{horizontal_tilt}")
    print("Done")


def main():
    source_directory = Path(__file__).resolve().parent
    calibration_directory = Path(sys.argv[1])

    print(f"Calibration directory is {calibration_directory}")

    target_file = calibration_directory / "merged_depth_results_all_2.csv"

    for cam in [10, 11]:
        for path in (calibration_directory / f"camera_{cam}").glob('*'):
            last_directory = path.name
            try:
                ground_truth = float(
                    ''.join([c for c in last_directory if c.isdigit() or c == '.']))
            except ValueError:
                print("Skipping {last_directory}")
                continue

            print(f"Last directory is {last_directory}")
            print(f"Ground truth is {ground_truth}")

            for position in ["center", "left", "right", "top", "bottom"]:
                for resolution in ["fullRes", "resizedRes"]:
                    print(f"Running in {cam} {path} m {position}")
                    depth_dir = path / position

                    if not depth_dir.is_dir():
                        print(f"Directory {depth_dir} does not exist")
                        continue

                    latest_directory = get_latest_directory(depth_dir)
                    if latest_directory is None:
                        print(
                            f"Directory {depth_dir} does not contain any subdirectories")
                        continue

                    if not latest_directory.is_dir():
                        print(f"Directory {latest_directory} does not exist")
                        continue

                    print(f"Base directory is {latest_directory}")

                    output_directory = latest_directory / resolution

                    for calibration_file in (calibration_directory / f"camera_{cam}" / "calibrations").glob('*'):
                        calibration_stem = calibration_file.stem
                        for orientation in ["vertical", "horizontal"]:
                            from_file_path = output_directory / calibration_stem / f"results_{orientation[:3]}_auto.txt"
                            merge_results(from_file_path, target_file, cam, orientation, resolution, position, calibration_stem)

if __name__ == "__main__":
    main()