#!/bin/bash
# =============================================================================
# entrypoint.sh
# Railway meng-inject $PORT secara otomatis saat container berjalan.
# Script ini membaca $PORT lalu meneruskannya ke tensorflow_model_server.
# Fallback: 8501 jika $PORT tidak di-set.
# =============================================================================

set -e

REST_PORT="${PORT:-8501}"

echo "================================================="
echo "  TensorFlow Serving — Mushroom Classification"
echo "================================================="
echo "  MODEL_NAME      : ${MODEL_NAME}"
echo "  MODEL_BASE_PATH : ${MODEL_BASE_PATH}"
echo "  REST PORT       : ${REST_PORT}"
echo "================================================="

# Verifikasi folder model ada
if [ ! -d "${MODEL_BASE_PATH}" ]; then
  echo "[ERROR] Folder model tidak ditemukan: ${MODEL_BASE_PATH}"
  exit 1
fi

# Cari subfolder versi numerik
VERSIONS=$(ls "${MODEL_BASE_PATH}" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
if [ -z "${VERSIONS}" ]; then
  echo "[ERROR] Tidak ada subfolder versi numerik di: ${MODEL_BASE_PATH}"
  echo "[INFO]  Isi folder: $(ls ${MODEL_BASE_PATH})"
  exit 1
fi

LATEST=$(echo "${VERSIONS}" | tail -1)
echo "[INFO]  Versi tersedia : ${VERSIONS}"
echo "[INFO]  Versi aktif    : ${LATEST}"

# Verifikasi saved_model.pb ada
if [ ! -f "${MODEL_BASE_PATH}/${LATEST}/saved_model.pb" ]; then
  echo "[ERROR] saved_model.pb tidak ada di: ${MODEL_BASE_PATH}/${LATEST}/"
  exit 1
fi

echo "[INFO]  saved_model.pb OK — memulai TF Serving..."

exec tensorflow_model_server \
  --port=8500 \
  --rest_api_port="${REST_PORT}" \
  --model_name="${MODEL_NAME}" \
  --model_base_path="${MODEL_BASE_PATH}" \
  --enable_model_warmup=false \
  --rest_api_timeout_in_ms=600000
