#!/bin/sh
set -eu

STATE_FILE="${ROUTER_STATE_FILE:-/state/rollout_state.json}"
HTTP_OUT="/etc/nginx/generated/rollout_http.conf"
PREDICT_OUT="/etc/nginx/generated/rollout_predict_location.conf"

mkdir -p /etc/nginx/generated
mkdir -p /state

python3 /opt/gateway/reloader.py \
  --state-file "$STATE_FILE" \
  --http-out "$HTTP_OUT" \
  --predict-out "$PREDICT_OUT" \
  --nginx-conf /etc/nginx/nginx.conf \
  --once

python3 /opt/gateway/reloader.py \
  --state-file "$STATE_FILE" \
  --http-out "$HTTP_OUT" \
  --predict-out "$PREDICT_OUT" \
  --nginx-conf /etc/nginx/nginx.conf \
  --watch &

exec nginx -g 'daemon off;'
