FROM tensorflow/serving:2.13.0

ENV MODEL_NAME=mushroom_model
ENV MODEL_BASE_PATH=/models/mushroom_model
ENV PORT=8501

COPY model/ /models/mushroom_model/

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]