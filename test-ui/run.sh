#!/usr/bin/env bash
# Launch the test UI server
# Usage: cd Research. && bash test-ui/run.sh

set -e
cd "$(dirname "$0")/.."
echo "Starting Test UI server..."
PYTHON="${PYTHON:-/opt/homebrew/bin/python3.11}"
$PYTHON test-ui/server.py
