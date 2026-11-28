#!/usr/bin/env bash
# Chuyển model pose của nhóm (YOLO11n-pose v5) sang .cvimodel INT8 cho MaixCAM, cắt đúng 3 output head mà
# MaixPy nn.YOLO11 cần (box DFL / điểm tin cậy / keypoint). Bản cũ chỉ có 1 output (output0) -> "get tensor idx error".
#
# Chạy trong WSL Ubuntu (không cần sudo):
#   bash /mnt/d/documentss/DoAnNganh/FallApp/fall_web/maixcam_app/convert_pose.sh
#
# Đầu vào (đã chuẩn bị sẵn trên Windows trong output/maix_convert/):
#   yolo11n_pose_f11_best_v5_maix.onnx   ONNX 320x224, tensor keypoint đã đổi tên thành /model.23/Concat_output_0
#   calib_images/ (200 ảnh từ video của nhóm), test.jpg
# Đầu ra: maixcam_app/model_classify/yolo11n_pose_f11_best_v5.cvimodel (bản cũ lưu ở output/maix_convert/old/)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/output/maix_convert"
NAME=yolo11n_pose_f11_best_v5
WORK="$HOME/tpu_work/$NAME"
VENV="$HOME/tpu_mlir_venv"
OUTS="/model.23/dfl/conv/Conv_output_0,/model.23/Sigmoid_output_0,/model.23/Concat_output_0"
SCALE=0.00392156862745098

echo "== 1/4 Công cụ: uv + Python 3.10 + tpu_mlir (lần đầu tải ~300 MB) =="
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if [ ! -x "$VENV/bin/python" ]; then
    uv python install 3.10
    uv venv -p 3.10 "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
# tpu_mlir còn "import pkg_resources" (thuộc setuptools; venv của uv không có sẵn, setuptools >= 81 đã bỏ nó)
python -c "import pkg_resources" 2>/dev/null || uv pip install "setuptools<81"
if ! python -c "import tpu_mlir" 2>/dev/null; then
    # tpu_mlir ghim opencv-python-headless==4.8.0.74 (đã bị gỡ khỏi PyPI, "use 4.8.0.76") -> ép dùng 4.8.0.76
    # và pycocotools==2.0.6 (không có wheel, phải biên dịch C, máy chưa có 'cc') -> 2.0.8 có wheel sẵn
    OVR="$(mktemp)"
    printf '%s\n' "opencv-python-headless==4.8.0.76" "pycocotools==2.0.8" > "$OVR"
    uv pip install "tpu_mlir[onnx]==1.30.2" --override "$OVR"
    rm -f "$OVR"
fi
# tpu_mlir dùng nhưng không khai báo: psutil (cả 3 công cụ), torch (run_calibration) -> torch bản CPU (~200 MB)
python -c "import psutil" 2>/dev/null || uv pip install psutil
python -c "import torch" 2>/dev/null || uv pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu
# tpu_mlir mang theo libc/libm/libpthread/libstdc++ của Ubuntu 22.04 (glibc 2.35). Trên Ubuntu mới hơn
# (WSL đang là 26.04, glibc 2.43) chúng xung đột với thư viện hệ thống -> "GLIBC_ABI_DT_RELR not found".
# Cất chúng sang _bundled_sys/ để dùng bản của hệ điều hành (tương thích ngược). Chỉ sửa trong venv.
TP="$(python -c 'import os, importlib.util as u; print(os.path.join(os.path.dirname(u.find_spec("tpu_mlir").origin), "lib", "third_party"))')"
# getconf in đúng 1 dòng "glibc 2.43" (không dùng "ldd --version | head": head đóng ống -> ldd lỗi -> set -e dừng im lặng)
GLIBC_MINOR="$( (getconf GNU_LIBC_VERSION 2>/dev/null || echo 'glibc 2.0') | awk '{split($2, v, "."); print v[2]}')"
echo "glibc hệ thống: 2.${GLIBC_MINOR}"
if [ "${GLIBC_MINOR:-0}" -gt 35 ] && [ -f "$TP/libc.so.6" ]; then
    echo "glibc hệ thống 2.$GLIBC_MINOR > 2.35: cất thư viện hệ thống mà tpu_mlir mang theo -> $TP/_bundled_sys/"
    mkdir -p "$TP/_bundled_sys"
    for f in libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 librt.so.1 libutil.so.1 libstdc++.so.6 libgcc_s.so.1; do
        [ -e "$TP/$f" ] && mv "$TP/$f" "$TP/_bundled_sys/"
    done
fi
pick() { command -v "$1.py" >/dev/null 2>&1 && echo "$1.py" || echo "$1"; }
TRANSFORM=$(pick model_transform); CALIB=$(pick run_calibration); DEPLOY=$(pick model_deploy)

echo "== 2/4 ONNX -> MLIR (cắt 3 output head) =="
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC/${NAME}_maix.onnx" "$SRC/calib_images" "$SRC/test.jpg" "$WORK/"
cd "$WORK"
$TRANSFORM --model_name "$NAME" --model_def "${NAME}_maix.onnx" --input_shapes "[[1,3,224,320]]" \
    --mean 0,0,0 --scale "$SCALE,$SCALE,$SCALE" --keep_aspect_ratio --pixel_format rgb \
    --output_names "$OUTS" --test_input test.jpg --test_result "${NAME}_top_outputs.npz" --mlir "$NAME.mlir"

echo "== 3/4 Hiệu chỉnh INT8 (200 ảnh) =="
$CALIB "$NAME.mlir" --dataset calib_images --input_num 200 -o "${NAME}_cali_table"

echo "== 4/4 MLIR -> cvimodel INT8 cho cv181x (MaixCAM) =="
$DEPLOY --mlir "$NAME.mlir" --quantize INT8 --quant_input --calibration_table "${NAME}_cali_table" \
    --processor cv181x --test_input "${NAME}_in_f32.npz" --test_reference "${NAME}_top_outputs.npz" \
    --tolerance 0.9,0.6 --model "$NAME.cvimodel"

DST="$ROOT/maixcam_app/model_classify"
mkdir -p "$SRC/old"
[ -f "$DST/$NAME.cvimodel" ] && cp "$DST/$NAME.cvimodel" "$SRC/old/$NAME.cvimodel.$(date +%Y%m%d_%H%M%S)"
cp "$NAME.cvimodel" "$DST/$NAME.cvimodel"
echo
echo "XONG: $DST/$NAME.cvimodel ($(du -h "$NAME.cvimodel" | cut -f1))"
echo "Tiếp theo: MaixVision -> Run Project lại. Log phải có 'OK: .../$NAME.mud (320x224)'."
