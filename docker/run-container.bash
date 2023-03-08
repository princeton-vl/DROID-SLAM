#!/bin/bash

# start sharing xhost
xhost +local:root

# run docker
docker run \
  --name droid-slam \
  --gpus all \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:$HOME/.Xauthority \
  -v $HOME/projects/uav_mapping/DROID-SLAM:/root/DROID-SLAM \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$HOME/.Xauthority \
  -e QT_X11_NO_MITSHM=1 \
  -it pytholic/droid-slam
