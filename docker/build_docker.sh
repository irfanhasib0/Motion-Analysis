#!/bin/bash
# Check if the 'docker' group exists and add the current user if not already a member.
# This is necessary because Docker commands require the user to be in the 'docker' group for permission.
#groups $USER | grep -q '\bdocker\b'; then
#sudo usermod -aG docker $USER
docker build --no-cache -t cv-env -f Dockerfile .
