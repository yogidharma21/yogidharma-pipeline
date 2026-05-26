FROM tensorflow/serving:2.13.0

LABEL maintainer="yogidharma21"
LABEL description="TF Serving — Mushroom Classification Model"
LABEL version="1.0"

ENV MODEL_NAME=mushroom_model
ENV MODEL_BASE_PATH=/models/mushroom_model
ENV PORT=8501

COPY serving/model/ /models/mushroom_model/
COPY serving/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]