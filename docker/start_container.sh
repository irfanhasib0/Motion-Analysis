#!/bin/bash
set -e
#xhost +
#docker stop cv-env || true
CONTAINER_NAME="cv-env"
IMAGE_NAME="cv-env"
WORKSPACE_DIR="/root/Motion-Analysis"

docker run -d --rm --gpus all \
    --device=/dev/dri:/dev/dri \
    --device=/dev/video0:/dev/video0 \
    --device=/dev/video1:/dev/video1 \
    --device=/dev/video2:/dev/video2 \
    --privileged \
    -v /home/irfan/Desktop/Code/Motion-Analysis:/root/Motion-Analysis \
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
    -w "$WORKSPACE_DIR/api/backend" \
    --name "$CONTAINER_NAME" "$IMAGE_NAME" \
    bash -c "jupyter lab --allow-root --ip=0.0.0.0 --port=8001 --LabApp.token='' --notebook-dir='$WORKSPACE_DIR'"

echo "Waiting for container to start..."

for i in {1..15}; do
    if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
        break
    fi
    sleep 1
done

if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "Container '$CONTAINER_NAME' is not running. Recent logs:"
    docker logs "$CONTAINER_NAME" || true
    exit 1
fi

docker exec -d "$CONTAINER_NAME" bash -c "cd $WORKSPACE_DIR/docker && bash install.sh"
