import os
import re
import sys
import argparse
import subprocess
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
    return Path(latest_dir)

def main():
    parser = argparse.ArgumentParser(description="Script to process camera data.")
    parser.add_argument("calibration_dir", help="Path to the calibration directory")
    parser.add_argument("-mode", help="Mode to run in", choices=["measure", "interactive"], default="interactive")
    args = parser.parse_args()

    source_directory = Path(__file__).resolve().parent
    calibration_directory = Path(args.calibration_dir)

    print(f"Calibration directory is {calibration_directory}")

    camera_names = [f"camera_{i}" for i in range(5, 7)]
    positions = ["center", "left", "right", "top", "bottom"]

    for camera in camera_names:
        for path in (calibration_directory / camera).glob('*'):
            path_basename = path.name

            if not path.is_dir() or not re.match(r'^\d+(\.\d+)?m$', path_basename):
                print(f"Skipping {path_basename}")
                continue

            ground_truth = float(re.findall(r'[+-]?\d+(?:\.\d+)?', path_basename)[0])
            print(f"Last directory is {path_basename}")
            print(f"Ground truth is {ground_truth}")

            for i, calibration_file in enumerate((calibration_directory / camera / 'calibrations').glob('*')):
                if i > 0 and args.mode == "interactive":
                    print("Skipping calibration file", calibration_file)
                    break
                for position in positions:
                    print(f"Running in {camera} {path} m {position}")
                    depth_dir = path / position
                    if not depth_dir.is_dir():
                        print(f"Directory {depth_dir} does not exist")
                        continue

                    latest_directory = get_latest_directory(depth_dir)
                    if latest_directory is None:
                        print(f"Directory {depth_dir} does not contain any subdirectories")
                        continue
                    if not latest_directory.is_dir():
                        print(f"Directory {latest_directory} does not exist")
                        continue

                    calibration_file_stem = calibration_file.stem

                    for resolution_type in ["fullRes", "resizedRes"]:
                        output_directory = latest_directory / resolution_type / calibration_file_stem
                        output_directory.mkdir(parents=True, exist_ok=True)

                        print(f"Base directory is {latest_directory}")
                        print(f"Output directory is {output_directory}")

                        for orientation, prefix in [("vertical", "ver"), ("horizontal", "hor")]:
                            print(f"Running {orientation}")
                            print(f"Running {position}")

                            cmd = [
                                sys.executable,
                                str(source_directory / "main.py"),
                                "-calib", str(latest_directory / "calib.json"),
                                "-depth", str(output_directory / f"{orientation}Depth.npy"),
                                "-rectified", str(output_directory / f"leftRectified{orientation.capitalize()}.npy"),
                                "-gt", str(ground_truth),
                                "-out_results_f", str(output_directory / f"results_{prefix}_auto.txt"),
                            ]
                            if args.mode == "interactive":
                                cmd += ["-mode", "interactive"]
                                cmd += ["-roi_file", str(latest_directory / resolution_type / f"roi_{prefix}.txt")]
                            else:
                                cmd +=["-mode", "measure"]
                                cmd +=["-set_roi_file", str(latest_directory / resolution_type / f"roi_{prefix}.txt")]
                            result = subprocess.run(cmd, capture_output=True)
                            if result.returncode != 0:
                                print(f"Error running the script: {result.stderr.decode('utf-8'), result.stdout.decode('utf-8')}")

if __name__ == "__main__":
    main()
