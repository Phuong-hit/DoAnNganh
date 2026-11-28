#!/usr/bin/env bash
# Đo lại 4 cvimodel của thí nghiệm gộp / không gộp (đã tạo bởi exp_npu_fusion.sh) trên các clip KHÔNG trùng.
set -eo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"; W="$HOME/tpu_work/fusion_exp"
. "$HOME/tpu_mlir_venv/bin/activate"; cd "$W"
python "$R/eval_cvimodel_npu.py" "$R/output/maix_convert/shiftgcn/eval.npz" \
  shiftgcn_npu/shiftgcn_npu_int8.cvimodel shiftgcn_npu/shiftgcn_npu_bf16.cvimodel \
  shiftgcn_unfused/shiftgcn_unfused_int8.cvimodel shiftgcn_unfused/shiftgcn_unfused_bf16.cvimodel --all --out choice.txt \
  2>&1 | grep -aE "cvimodel: |Bỏ|CHỌN" | sed "s/.*version: [0-9.]*//" | tee "$R/output/bench/npu_fusion_exp.txt"
