#!/usr/bin/env bash
set -euo pipefail

# Push H.264 from a V4L2 camera to an RTSP server.
# Robustly handles input pixel formats and chooses a working encoder.

DEVICE=${DEVICE:-/dev/video0}
RTSP_URL=${RTSP_URL:-rtsp://127.0.0.1:8554/camera1}
FRAMERATE=${FRAMERATE:-30}
VIDEO_SIZE=${VIDEO_SIZE:-}

echo "Device: ${DEVICE} | RTSP: ${RTSP_URL} | FPS: ${FRAMERATE}"

detect_input_format() {
	if command -v v4l2-ctl >/dev/null 2>&1; then
		if v4l2-ctl --device "${DEVICE}" --list-formats-ext 2>/dev/null | grep -q "MJPG"; then
			echo mjpeg
			return
		fi
		if v4l2-ctl --device "${DEVICE}" --list-formats-ext 2>/dev/null | grep -q "YUYV"; then
			echo yuyv422
			return
		fi
	fi
	echo yuyv422
}

detect_video_size() {
	if [ -n "${VIDEO_SIZE}" ]; then
		echo "${VIDEO_SIZE}"
		return
	fi
	if command -v v4l2-ctl >/dev/null 2>&1; then
		# Pick the first even resolution listed, prefer >= 640x480
		local sz
		sz=$(v4l2-ctl --device "${DEVICE}" --list-formats-ext 2>/dev/null | awk '/Size: Discrete/ {print $3}' | sed 's/x/*/g' | sort -t'*' -k1,1n -k2,2n | sed 's/*/x/g' | tail -1)
		if [[ -n "${sz}" ]]; then
			echo "${sz}"
			return
		fi
	fi
	echo 640x480
}

INPUT_FORMAT=$(detect_input_format)
VIDEO_SIZE=$(detect_video_size)
echo "Detected input format: ${INPUT_FORMAT}, size: ${VIDEO_SIZE}"

choose_encoder() {
	if gst-inspect-1.0 v4l2h264enc >/dev/null 2>&1; then
		echo "v4l2h264enc extra-controls=\"controls,video_bitrate=1000000;\""
	elif gst-inspect-1.0 vaapih264enc >/dev/null 2>&1; then
		echo "vaapih264enc rate-control=cbr bitrate=1000 keyframe-period=30"
	elif gst-inspect-1.0 x264enc >/dev/null 2>&1; then
		echo "x264enc tune=zerolatency speed-preset=ultrafast bitrate=1000 key-int-max=${FRAMERATE}"
	else
		echo "Error: No H.264 encoder found (v4l2h264enc/vaapih264enc/x264enc)." >&2
		exit 1
	fi
}

ENCODER=$(choose_encoder)
echo "Using encoder: ${ENCODER}"

if gst-inspect-1.0 rtspclientsink >/dev/null 2>&1; then
	echo "Using GStreamer rtspclientsink to push to ${RTSP_URL}"
	if v4l2-ctl --device "${DEVICE}" --list-formats-ext 2>/dev/null | grep -q "MJPG"; then
		# MJPEG needs jpegdec to get raw video
		gst-launch-1.0 -e -v \
			v4l2src device="${DEVICE}" ! image/jpeg,framerate=${FRAMERATE}/1 ! jpegdec ! \
			videoconvert ! video/x-raw,format=I420,width=$(echo ${VIDEO_SIZE} | cut -dx -f1),height=$(echo ${VIDEO_SIZE} | cut -dx -f2) ! \
			${ENCODER} ! h264parse config-interval=1 ! \
			rtspclientsink location="${RTSP_URL}" protocols=tcp
	else
		gst-launch-1.0 -e -v \
			v4l2src device="${DEVICE}" ! video/x-raw,format=I420,width=$(echo ${VIDEO_SIZE} | cut -dx -f1),height=$(echo ${VIDEO_SIZE} | cut -dx -f2),framerate=${FRAMERATE}/1 ! \
			videoconvert ! ${ENCODER} ! h264parse config-interval=1 ! \
			rtspclientsink location="${RTSP_URL}" protocols=tcp
	fi
else
	echo "rtspclientsink not found. Falling back to FFmpeg if available..."
	if command -v ffmpeg >/dev/null 2>&1; then
		echo "Using FFmpeg to push RTSP to ${RTSP_URL}"
		# Force correct V4L2 input format and size; ensure even dimensions and YUV420P for H.264
		ffmpeg -hide_banner -nostdin -v verbose \
			-re -f v4l2 -input_format ${INPUT_FORMAT} -video_size ${VIDEO_SIZE} -framerate ${FRAMERATE} -i "${DEVICE}" \
			-vf "format=yuv420p" -pix_fmt yuv420p \
			-c:v libx264 -preset veryfast -tune zerolatency -b:v 1M -g ${FRAMERATE} -keyint_min ${FRAMERATE} -sc_threshold 0 \
			-f rtsp -rtsp_transport tcp -rtsp_flags prefer_tcp -muxdelay 0 -muxpreload 0 "${RTSP_URL}"
	else
		echo "Error: Neither GStreamer rtspclientsink nor FFmpeg is available."
		echo "Install GStreamer bad/ugly/libav plugins or FFmpeg. Examples:"
		echo "  sudo apt install -y v4l-utils ffmpeg"
		echo "  sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good"
		echo "  sudo apt install -y gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-vaapi"
		exit 1
	fi
fi