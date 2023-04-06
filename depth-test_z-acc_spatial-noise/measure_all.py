import os
import re
import sys
from pathlib import Path

def get_latest_id(parent_dir: Path) -> Path:
    return max((d for d in parent_dir.iterdir() if d.is_dir()), key=lambda d: int(re.findall(r'\d+', d.name)[0]))

def main():
    source_dir = Path(__file__).resolve().parent
    calib_dir = Path(sys.argv[1])

    print(f"Calib dir is {calib_dir}")

    for cam in (f"camera_{i}" for i in range(5, 7)):
        for path in (calib_dir / cam).glob('*'):
            last_directory = path.name
            gt = float(re.findall(r'[+-]?\d+(?:\.\d+)?', last_directory)[0])
            print(f"Last directory is {last_directory}")
            print(f"GT is {gt}")

            for position in ['center']:
                for resolution in ['fullRes', 'resizedRes']:
                    print(f"Running in {cam} {path} m {position}")
                    dDir = path / position
                    if not dDir.is_dir():
                        print(f"Directory {dDir} does not exist")
                        continue

                    latestDir = get_latest_id(dDir)
                    if not latestDir.is_dir():
                        print(f"Directory {latestDir} does not exist")
                        continue

                    outDir = latestDir / resolution
                    print(f"Base dir is {latestDir}")
                    print(f"Out dir is {outDir}")

                    for orientation, prefix in [("vertical", "ver"), ("horizontal", "hor")]:
                        print(f"Running {orientation}")
                        print(f"Running {position}")

                        args = [
                            sys.executable,
                            str(source_dir / "main.py"),
                            "-mode", "measure",
                            "-calib", str(latestDir / "calib.json"),
                            "-depth", str(outDir / f"{orientation}Depth.npy"),
                            "-rectified", str(outDir / f"leftRectified{orientation.capitalize()}.npy"),
                            "-gt", str(gt),
                            "-out_results_f", str(outDir / f"results_{prefix}_auto.txt"),
                            "-set_roi_file", str(outDir / f"roi_{orientation}.txt"),
                        ]
                        os.system(" ".join(args))

if __name__ == "__main__":
    main()