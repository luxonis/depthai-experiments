# Read the CSV file
# Take the last row
# Genereate a new and ID
# Add it to the global csv file

from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(prog = "Merge result files")

parser.add_argument('--pathFrom', type=str, required=True, default=None, help="Path to the folder containing the results to merge")
parser.add_argument('--pathTo', type=str, required=True, default=None, help="Path to the folder containing the results to merge")
parser.add_argument('--type', type=str, required=True, default=None, help="Vertical or horizontal")
parser.add_argument('--camId', type=int, required=True, default=None, help="Camera ID")
parser.add_argument('--side', type=str, required=True, default=None, help="Left or right or center")
parser.add_argument('--resolution', type=str, required=True, default=None, help="Resolution of the camera")

args = parser.parse_args()
fromFile = Path(args.pathFrom)
toFile = Path(args.pathTo)

if not toFile.exists():
    with open(toFile, "w") as f:
        f.write("cam_id,type,resolution,side,ground_truth,z_measured_mean,z_measured_median,z_accuracy,spatial_noise,subpixel_spatial_noise,vertical_tilt,horizontal_tilt")

if not fromFile.exists():
    print("Skipping file:", fromFile.absolute(), "because it does not exist")
    exit(0)

fromFileSplit = fromFile.read_text().splitlines()[-1].split(",")
# Looks like this: unknown,4.0,4463.098320701224,4463.098573606749,11.62,463.13,0.97,-0.05069701339945318,0.39542779453497495

ground_truth = fromFileSplit[1]
z_measured_mean = fromFileSplit[2]
z_measured_meadian = fromFileSplit[3]
z_accuracy = fromFileSplit[4]
space_noise = fromFileSplit[5]
subpixel_space_noise = fromFileSplit[6]
vertical_tilt = fromFileSplit[7]
horizontal_tilt = fromFileSplit[8]

with open(toFile, "a") as f:
    f.write(f"\n{args.camId},{args.type},{args.resolution},{args.side},{ground_truth},{z_measured_mean},{z_measured_meadian},{z_accuracy},{space_noise},{subpixel_space_noise},{vertical_tilt},{horizontal_tilt}")