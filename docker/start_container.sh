#!/bin/bash
set -e
xhost +
docker stop cv-env || true
docker run -d --rm --gpus all \
    --device=/dev/dri:/dev/dri \
    -v /home/irfan/Desktop/Code/:/projects/ \
    -v /media:/media \
    -v /home/irfan/Desktop/Data/:/data/ \
    -v /home/irfan/.Xauthority:/root/.Xauthority:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e LIBGL_ALWAYS_INDIRECT=1 \
    -e MESA_GL_VERSION_OVERRIDE=3.3 \
    --net=host \
    -p 8001:8001 \
    -w /projects/Motion-Analysis/ \
    --name cv-env cv-env bash -c "jupyter lab --allow-root --ip=0.0.0.0 --port=8001 --LabApp.token='' --notebook-dir='/projects'"
