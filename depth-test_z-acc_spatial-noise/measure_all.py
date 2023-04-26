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

def read_checkpoint():
    checkpoint_file = '.checkpoint_measure.txt'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return f.read().strip()
    return ''

def save_checkpoint(camera, path, calibration_file, position):
    checkpoint_file = '.checkpoint_measure.txt'
    with open(checkpoint_file, 'w') as f:
        f.write(f"{camera},{path},{calibration_file},{position}")

def main():
    parser = argparse.ArgumentParser(description="Script to process camera data.")
    parser.add_argument("calibration_dir", help="Path to the calibration directory")
    parser.add_argument("-mode", help="Mode to run in", choices=["measure", "interactive"], default="interactive")
    parser.add_argument("--continue", dest="continue_from_last", action="store_true", help="Continue from the last checkpoint")
    parser.add_argument("--camera_ids", required=True, help="List of cameras to process", default=[], nargs='+', type=int)
    parser.add_argument("--positions", required=False, help="List of positions to process", default=["left", "right", "center", "top", "bottom"], nargs='+', type=str)
    args = parser.parse_args()

    source_directory = Path(__file__).resolve().parent
    calibration_directory = Path(args.calibration_dir)

    print(f"Calibration directory is {calibration_directory}")

    camera_names = [f"camera_{i}" for i in args.camera_ids]
    positions = args.positions

    last_checkpoint = read_checkpoint()
    if args.continue_from_last and last_checkpoint:
        last_camera, last_path, last_calibration_file, last_position = last_checkpoint.split(',')

    start_from_last = args.continue_from_last and last_checkpoint != ''

    try:
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
                    print(calibration_file.name)
                    # if "calib.json" not in calibration_file.name:
                    #     continue
                    for position in positions:
                        if start_from_last:
                            if (camera, path, calibration_file, position) != (last_camera, last_path, last_calibration_file, last_position):
                                continue
                            else:
                                start_from_last = False

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

                        for resolution_type in ["depthOCV"]:
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
                                    "-rs", str(0.2),
                                ]
                                if args.mode == "interactive":
                                    cmd += ["-mode", "interactive"]
                                    cmd += ["-roi_file", str(latest_directory / resolution_type / f"roi_{prefix}.txt")]
                                else:
                                    cmd +=["-mode", "measure"]
                                    cmd +=["-set_roi_file", str(latest_directory / resolution_type / f"roi_{prefix}.txt")]
                                print("Hello")
                                print(" ".join(cmd))
                                result = subprocess.run(cmd, capture_output=False)
                                print("Hello2")
                                if result.returncode != 0:
                                    command = " ".join(cmd)
                                    print(command)
                                    if result.stderr and result.stdout:
                                       print(f"Error running the script: {result.stderr.decode('utf-8'), result.stdout.decode('utf-8')}")
                                save_checkpoint(camera, path, calibration_file, position)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")
        save_checkpoint(camera, path, calibration_file, position)
        print("Progress saved. Exiting.")
        exit(1)

if __name__ == "__main__":
    main()
