#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# docker_commands.sh
# Kumpulan perintah Docker untuk build, run, dan test TF Serving container.
#
# Jalankan dari root project:
#   bash serving/docker_commands.sh
# Atau eksekusi tiap bagian secara manual sesuai kebutuhan.
# ─────────────────────────────────────────────────────────────────────────────

set -e  # hentikan jika ada perintah yang gagal

# ── Variabel ──────────────────────────────────────────────────────────────────
IMAGE_NAME="mushroom-serving"
IMAGE_TAG="1.0"
CONTAINER_NAME="mushroom-tf-serving"
REST_PORT=8501
GRPC_PORT=8500

# Jalur model hasil pipeline TFX (sesuaikan jika berbeda)
# Format: serving_model_dir/mushroom_model/<timestamp>/
# Salin versi terbaru ke serving/model/1/
MODEL_SRC="serving_model_dir/mushroom_model"
MODEL_DST="serving/model/1"

# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 1 — Salin model dari direktori serving ke folder Docker build context
# ─────────────────────────────────────────────────────────────────────────────
echo "📦 Menyalin model ke serving/model/1/ ..."

# Dapatkan subfolder timestamp terbaru dari Pusher output
LATEST_VERSION=$(ls -t "$MODEL_SRC" | head -1)
if [ -z "$LATEST_VERSION" ]; then
    echo "❌ ERROR: Tidak ada model di $MODEL_SRC"
    echo "   Pastikan pipeline sudah dijalankan dan model sudah di-push."
    exit 1
fi

mkdir -p "$MODEL_DST"
cp -r "$MODEL_SRC/$LATEST_VERSION/." "$MODEL_DST/"
echo "✅ Model disalin dari $MODEL_SRC/$LATEST_VERSION ke $MODEL_DST"

# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 2 — Build Docker image
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "🔨 Building Docker image: $IMAGE_NAME:$IMAGE_TAG ..."
docker build \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    -f serving/Dockerfile \
    serving/

echo "✅ Image berhasil di-build: $IMAGE_NAME:$IMAGE_TAG"

# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 3 — Hapus container lama jika ada
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "🧹 Menghapus container lama (jika ada) ..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 4 — Jalankan container
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "🚀 Menjalankan container: $CONTAINER_NAME ..."
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$REST_PORT:8501" \
    -p "$GRPC_PORT:8500" \
    "$IMAGE_NAME:$IMAGE_TAG"

echo "✅ Container berjalan di latar belakang."
echo ""
echo "   REST API : http://localhost:$REST_PORT/v1/models/mushroom_model"
echo "   Health   : http://localhost:$REST_PORT/v1/models/mushroom_model/versions/1"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 5 — Tunggu server siap dan verifikasi
# ─────────────────────────────────────────────────────────────────────────────
echo "⏳ Menunggu server siap (10 detik) ..."
sleep 10

echo "🔍 Cek status model:"
curl -s "http://localhost:$REST_PORT/v1/models/mushroom_model" | python3 -m json.tool || true

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "📋 Perintah berguna lainnya:"
echo ""
echo "  Lihat log container    : docker logs -f $CONTAINER_NAME"
echo "  Hentikan container     : docker stop $CONTAINER_NAME"
echo "  Hapus container        : docker rm $CONTAINER_NAME"
echo "  Hapus image            : docker rmi $IMAGE_NAME:$IMAGE_TAG"
echo "  Masuk ke container     : docker exec -it $CONTAINER_NAME bash"
echo "──────────────────────────────────────────────────────────────"
