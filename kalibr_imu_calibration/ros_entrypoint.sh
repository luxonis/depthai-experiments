#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash" 
source "/catkin_ws/devel/setup.bash" 
source "/home/dai_ws/devel/setup.bash"
export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
exec "$@"