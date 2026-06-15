#!/usr/bin/env bash
# Runs ON the box. Manages the ComfyUI server via a pidfile (so we never have to
# pkill by pattern — which can match and kill the controlling shell).
#   comfy.sh start | stop | restart | status | wait | logs [n]
set -euo pipefail
COMFY_DIR="${COMFY_DIR:-/ephemeral/ComfyUI}"
PORT="${COMFY_PORT:-8188}"
PIDFILE=/tmp/comfy.pid
LOG=/tmp/comfy.log

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-}" in
  start)
    if running; then echo "already running pid $(cat "$PIDFILE")"; exit 0; fi
    cd "$COMFY_DIR"
    nohup venv/bin/python main.py --listen 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    echo "started pid $(cat "$PIDFILE")"
    ;;
  stop)
    if running; then kill "$(cat "$PIDFILE")" && echo "stopped"; else echo "not running"; fi
    rm -f "$PIDFILE"
    ;;
  restart) "$0" stop || true; sleep 2; "$0" start ;;
  status)
    running && echo "running pid $(cat "$PIDFILE")" || echo "stopped"
    curl -s -o /dev/null -w "http %{http_code}\n" "http://127.0.0.1:$PORT/" || true
    ;;
  wait)
    for i in $(seq 1 90); do
      [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" 2>/dev/null)" = 200 ] \
        && { echo "UP ~$((i*2))s"; exit 0; }
      sleep 2
    done
    echo "did not come up; log tail:"; tail -25 "$LOG"; exit 1
    ;;
  logs) tail -"${2:-30}" "$LOG" ;;
  *) echo "usage: comfy.sh {start|stop|restart|status|wait|logs [n]}" >&2; exit 1 ;;
esac
