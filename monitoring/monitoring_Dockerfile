# =============================================================================
# Dockerfile — Prometheus Monitoring untuk Mushroom Classification
#
# Base image : prom/prometheus:v2.45.0
# Port       : 9090
# Config     : prometheus.yml
# =============================================================================

FROM prom/prometheus:v2.45.0

LABEL maintainer="yogidharma21"
LABEL description="Prometheus monitoring untuk Mushroom TF Serving"
LABEL version="1.0"

# Salin konfigurasi Prometheus ke dalam container
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Expose port Prometheus
EXPOSE 9090

# Jalankan Prometheus dengan konfigurasi yang sudah disalin
CMD [ \
    "--config.file=/etc/prometheus/prometheus.yml", \
    "--storage.tsdb.path=/prometheus", \
    "--web.console.libraries=/usr/share/prometheus/console_libraries", \
    "--web.console.templates=/usr/share/prometheus/consoles", \
    "--web.enable-lifecycle" \
]
