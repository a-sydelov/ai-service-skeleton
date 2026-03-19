FROM nginx:1.27-alpine

RUN apk add --no-cache python3

COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY gateway/entrypoint.sh /entrypoint.sh
COPY gateway/reloader.py /opt/gateway/reloader.py

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
