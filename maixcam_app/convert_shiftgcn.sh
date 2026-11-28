#!/usr/bin/env bash
# Chuyển Shift-GCN (bản viết lại cho NPU, src/shiftgcn_npu.py) sang .cvimodel cho NPU của MaixCAM.
# Tạo 2 bản: INT8 (nhanh nhất) và BF16 (chính xác hơn), đo cả hai trên 600 clip thật (mô phỏng NPU trên PC),
# rồi chép bản được chọn vào maixcam_app/model_classify/shiftgcn_npu.cvimodel (+ .mud).
#
# Trước đó trên Windows:  python export_maixcam_model.py --npu      (tạo output/maix_convert/shiftgcn/)
# Chạy trong WSL Ubuntu-22.04 (cùng môi trường với convert_pose.sh):
#   bash /mnt/d/documentss/DoAnNganh/FallApp/fall_web/maixcam_app/convert_shiftgcn.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/output/maix_convert/shiftgcn"
WORK="$HOME/tpu_work/shiftgcn"
NAME=shiftgcn_npu
. "$HOME/tpu_mlir_venv/bin/activate"
pick() { command -v "$1.py" >/dev/null 2>&1 && echo "$1.py" || echo "$1"; }
TRANSFORM=$(pick model_transform); CALIB=$(pick run_calibration); DEPLOY=$(pick model_deploy)

rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC/$NAME.onnx" "$SRC/calib" "$SRC/test.npz" "$SRC/eval.npz" "$WORK/"
cd "$WORK"

echo "== 1/4 ONNX -> MLIR =="
$TRANSFORM --model_name "$NAME" --model_def "$NAME.onnx" --input_shapes "[[1,3,17,17]]" \
    --test_input test.npz --test_result "${NAME}_top_outputs.npz" --mlir "$NAME.mlir"

echo "== 2/4 Hiệu chỉnh INT8 (200 clip thật) =="
$CALIB "$NAME.mlir" --dataset calib --input_num 200 -o "${NAME}_cali_table"

echo "== 3/4 MLIR -> cvimodel (INT8 và BF16, cv181x) =="
# Không dùng --test_reference ở đây: một bản lệch tolerance không làm dừng script; bước 4 đo kỹ hơn trên 600 clip.
MODELS=()
$DEPLOY --mlir "$NAME.mlir" --quantize INT8 --calibration_table "${NAME}_cali_table" --processor cv181x \
    --model "${NAME}_int8.cvimodel" && MODELS+=("${NAME}_int8.cvimodel") || echo "[!] Không tạo được bản INT8"
$DEPLOY --mlir "$NAME.mlir" --quantize BF16 --processor cv181x \
    --model "${NAME}_bf16.cvimodel" && MODELS+=("${NAME}_bf16.cvimodel") || echo "[!] Không tạo được bản BF16"
[ ${#MODELS[@]} -gt 0 ] || { echo "LỖI: không tạo được cvimodel nào"; exit 1; }

echo "== 4/4 Đo trên 600 clip (mô phỏng NPU) và chọn bản =="
python "$ROOT/eval_cvimodel_npu.py" eval.npz "${MODELS[@]}" --out choice.txt
CHOSEN="$(cat choice.txt)"

DST="$ROOT/maixcam_app/model_classify"
cp "$CHOSEN" "$DST/$NAME.cvimodel"
printf '[basic]\ntype = cvimodel\nmodel = %s.cvimodel\n' "$NAME" \
    > "$DST/$NAME.mud"
echo
echo "XONG: $DST/$NAME.cvimodel  (bản: $CHOSEN, $(du -h "$CHOSEN" | cut -f1))"
