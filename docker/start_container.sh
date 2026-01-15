#!/bin/bash
set -e
xhost +
docker stop cv-env || true
docker run -d --rm --gpus all \
    --device=/dev/dri:/dev/dri \
    --device=/dev/video0:/dev/video0 \
    --device=/dev/video1:/dev/video1 \
    --device=/dev/video2:/dev/video2 \
    --privileged \
    -v /home/irfan/Desktop/Code/:/projects/ \
    -v /media:/media \
    -v /home/irfan/Desktop/Data/:/data/ \
    -v /home/irfan/.Xauthority:/root/.Xauthority:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/shm:/dev/shm:rw \
    -e DISPLAY=$DISPLAY \
    -e LIBGL_ALWAYS_INDIRECT=1 \
    -e MESA_GL_VERSION_OVERRIDE=3.3 \
    --net=host \
    -p 3001:3001 \
    -p 8001:8001 \
    -p 9001:9001 \
    -w /projects/Motion-Analysis/ \
    --name cv-env cv-env bash -c "jupyter lab --allow-root --ip=0.0.0.0 --port=8001 --LabApp.token='' --notebook-dir='/projects'"
