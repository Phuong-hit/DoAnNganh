#!/usr/bin/env bash
# Thí nghiệm gộp / không gộp GCN: chuyển cả 2 bản Shift-GCN sang cvimodel INT8 + BF16 (cv181x), đo đổi nhãn trên
# cùng 600 clip (mô phỏng NPU). KHÔNG chép gì vào maixcam_app (chỉ để báo cáo).
# Trước đó trên Windows:  python export_maixcam_model.py --npu   và   python exp_npu_fusion.py
#   bash /mnt/d/documentss/DoAnNganh/FallApp/fall_web/maixcam_app/exp_npu_fusion.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$HOME/tpu_work/fusion_exp"
. "$HOME/tpu_mlir_venv/bin/activate"
pick() { command -v "$1.py" >/dev/null 2>&1 && echo "$1.py" || echo "$1"; }
TRANSFORM=$(pick model_transform); CALIB=$(pick run_calibration); DEPLOY=$(pick model_deploy)
mkdir -p "$WORK"
MODELS=()
for V in ${VARIANTS:-shiftgcn:shiftgcn_npu shiftgcn_unfused:shiftgcn_unfused}; do
  DIR=${V%%:*}; NAME=${V##*:}
  rm -rf "$WORK/$NAME"; mkdir -p "$WORK/$NAME"; cd "$WORK/$NAME"
  cp -r "$ROOT/output/maix_convert/$DIR/$NAME.onnx" "$ROOT/output/maix_convert/$DIR/calib" \
        "$ROOT/output/maix_convert/$DIR/test.npz" .
  echo "== $NAME: ONNX -> MLIR =="
  $TRANSFORM --model_name "$NAME" --model_def "$NAME.onnx" --input_shapes "[[1,3,17,17]]" \
      --test_input test.npz --test_result top.npz --mlir "$NAME.mlir"
  echo "== $NAME: hiệu chỉnh INT8 =="
  $CALIB "$NAME.mlir" --dataset calib --input_num 200 -o cali_table
  for Q in INT8 BF16; do
    q=$(echo $Q | tr A-Z a-z)
    ARGS=(--mlir "$NAME.mlir" --quantize $Q --processor cv181x --model "${NAME}_$q.cvimodel")
    [ $Q = INT8 ] && ARGS+=(--calibration_table cali_table)
    $DEPLOY "${ARGS[@]}" && MODELS+=("$WORK/$NAME/${NAME}_$q.cvimodel") || echo "[!] $NAME $Q: không tạo được"
  done
done
cd "$WORK"
echo; echo "== Kích thước =="; du -h "${MODELS[@]}"
echo; echo "== Đo trên 600 clip =="
python "$ROOT/eval_cvimodel_npu.py" "$ROOT/output/maix_convert/shiftgcn/eval.npz" "${MODELS[@]}" --all --out choice.txt \
  | tee -a "$ROOT/output/bench/npu_fusion_exp.txt"
