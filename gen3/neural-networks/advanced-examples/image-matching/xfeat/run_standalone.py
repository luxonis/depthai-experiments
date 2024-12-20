import argparse
import os
import shutil
import subprocess
import sys

# Define the oakapp file
OAKAPP_TEMP = 'oak_standalone.toml'
OAKAPP = 'oakapp.toml'

# Parse arguments
parser = argparse.ArgumentParser(description='Run standalone app.')
parser.add_argument('--model', required=True, help='Model to use.')
parser.add_argument('--device', required=True, help='Device to connect to.')
parser.add_argument('--fps_limit', help='FPS limit.')

args = parser.parse_args()

print(f"MODEL: {args.model}")
print(f"DEVICE: {args.device}")
print(f"FPS_LIMIT: {args.fps_limit}")

# Ensure the oakapp file exists
if not os.path.isfile(OAKAPP_TEMP):
    print(f"File not found: {OAKAPP_TEMP}")
    sys.exit(1)

# Create a temporary oakapp file
shutil.copy(OAKAPP_TEMP, OAKAPP)

# Use sed to replace placeholders with the provided values
with open(OAKAPP, 'r') as file:
    filedata = file.read()

filedata = filedata.replace('<Model>', f'--model {args.model}')

if args.fps_limit:
    filedata = filedata.replace('<FPS>', f'--fps_limit {args.fps_limit}')
else:
    filedata = filedata.replace('<FPS>', '')

with open(OAKAPP, 'w') as file:
    file.write(filedata)

# Connect to the device
if args.device:
    subprocess.run(['oakctl', 'connect', args.device])
else:
    subprocess.run(['oakctl', 'connect'])

# Run the example
subprocess.run(['oakctl', 'app', 'run', '.'])

# Remove the temporary oakapp file
os.remove(OAKAPP)