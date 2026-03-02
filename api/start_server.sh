#!/usr/bin/env bash
set -euo pipefail

# Root of the api folder (this script lives in api/)
API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$API_DIR/backend"
FRONTEND_DIR="$API_DIR/frontend"
LOG_FILE="$API_DIR/nvr.log"
LOGROTATE_CONF="/etc/logrotate.d/nvr"
RUN_USER="${USER:-$(id -un)}"
RUN_GROUP="${GROUP:-$(id -gn)}"

if command -v sudo >/dev/null 2>&1; then
    SUDO_CMD="sudo"
else
    SUDO_CMD=""
fi

if [ ! -d "$FRONTEND_DIR" ] || [ ! -d "$BACKEND_DIR" ]; then
    echo "Expected directories not found under API_DIR: $API_DIR"
    exit 1
fi

echo "Building frontend..."
cd "$FRONTEND_DIR"
if command -v npm >/dev/null 2>&1; then
    npm run build
else
    echo "Warning: 'npm' not found. Skipping frontend build step."
fi

echo "Stopping existing backend process (if any)..."
pkill -f "python3 start_server.py" || true

echo "Starting backend and writing logs to: $LOG_FILE"
cd "$BACKEND_DIR"
nohup python3 start_server.py >> "$LOG_FILE" 2>&1 &

echo "Installing daily logrotate config at $LOGROTATE_CONF"
if [ -n "$SUDO_CMD" ]; then
    sudo tee "$LOGROTATE_CONF" > /dev/null <<EOF
$LOG_FILE {
    su $RUN_USER $RUN_GROUP
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    create 0644 $RUN_USER $RUN_GROUP
}
EOF
else
    tee "$LOGROTATE_CONF" > /dev/null <<EOF
$LOG_FILE {
    su $RUN_USER $RUN_GROUP
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    create 0644 $RUN_USER $RUN_GROUP
}
EOF
fi

if command -v logrotate >/dev/null 2>&1; then
    echo "Validating logrotate config..."
    if [ -n "$SUDO_CMD" ]; then
        sudo logrotate -d "$LOGROTATE_CONF" >/dev/null
    else
        logrotate -d "$LOGROTATE_CONF" >/dev/null
    fi
else
    echo "Warning: 'logrotate' command not found. Config file was written to $LOGROTATE_CONF."
    echo "Install logrotate to enable daily rotation scheduling/validation:"
    echo "  Ubuntu/Debian: sudo apt-get install -y logrotate"
    echo "  Fedora/RHEL:   sudo dnf install -y logrotate"
    echo "  Arch:          sudo pacman -S logrotate"
fi

echo "Done. Backend logs: $LOG_FILE"
tail -f "$LOG_FILE"