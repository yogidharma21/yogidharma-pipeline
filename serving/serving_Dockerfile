# =============================================================================
# Dockerfile — TensorFlow Serving untuk Mushroom Classification Model
#
# Base image  : tensorflow/serving:2.12.0 (sesuai TF 2.12 di requirements.txt)
# Model name  : mushroom_model
# REST port   : dinamis dari Railway ($PORT), default fallback 8501
# Model path  : /models/mushroom_model/1/saved_model.pb
# =============================================================================

FROM tensorflow/serving:2.12.0

LABEL maintainer="yogidharma21"
LABEL description="TF Serving — Mushroom Classification Model"
LABEL version="1.0"

# Variabel environment
ENV MODEL_NAME=mushroom_model
ENV MODEL_BASE_PATH=/models/mushroom_model
ENV PORT=8501

# Salin model ke dalam container
# Struktur host:
#   serving/model/
#     1/
#       saved_model.pb
#       keras_metadata.pb
#       variables/
#       assets/
COPY model/ /models/mushroom_model/

# Salin entrypoint script untuk handle PORT Railway dinamis
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port REST API
EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
