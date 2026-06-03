#!/bin/bash

# Simple script to serve the website from this Mac
# Usage: ./serve.sh [port]

PORT=${1:-8080}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "🚀 Starting web server for: $DIR"
echo "   Local:    http://localhost:$PORT"
echo "   Network:  http://<LAN_IP>:$PORT"
echo "   Tailscale: http://<TAILSCALE_IP>:$PORT"
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd "$DIR"
