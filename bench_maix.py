#!/usr/bin/env python3
"""Bảng hiệu năng ĐO TRÊN MaixCAM (cùng kiểu bảng với bench_report.py).

MaixCAM (maixcam_app/main.py) tự đo mỗi khung: đọc camera, pose trên NPU, nén JPEG, vẽ màn hình, cả vòng,
và RAM (/proc). Script này: reset bộ đo trên board -> chờ --seconds -> lấy GET /bench -> in bảng + lưu file.

Chuẩn bị:
  1. MaixVision -> mở thư mục maixcam_app -> Run Project (nạp code mới có /bench).
  2. TẮT web / mọi trình duyệt đang xem luồng của MaixCAM (có người xem thì board phải nén JPEG -> chậm hơn).
  3. Đứng / đi lại trước camera trong lúc đo (pose chỉ tốn thời gian thật khi có người).

  python bench_maix.py --ip 10.123.219.1                     # đo 120 s
  python bench_maix.py --ip 10.123.219.1 --seconds 300 --label maix_v5
  python bench_maix.py --ip 10.123.219.1 --compare output/bench/report_cpu_shiftgcn_*.json   # thêm cột PC

Ra: output/bench/maix_<label>_<thời gian>.md / .tex / .json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.config import resolve  # noqa: E402


def get(ip, path, timeout=5):
    with urllib.request.urlopen(f"http://{ip}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


def pt_params(model_file: str | None) -> int | None:
    """Số tham số lấy từ file .pt cùng tên trong model_yolo/ (.cvimodel không lưu số tham số)."""
    if not model_file:
        return None
    pt = resolve("model_yolo") / (Path(model_file).stem + ".pt")
    if not pt.is_file():
        return None
    try:
        from ultralytics import YOLO
        return int(sum(p.numel() for p in YOLO(str(pt)).model.parameters()))
    except Exception:
        return None


def fmt(v, unit="", nd=2):
    if v is None:
        return "—"
    if isinstance(v, int):
        return f"{v:,}{unit}"
    return f"{v:.{nd}f}{unit}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", required=True, help="IP MaixCAM, ví dụ 10.123.219.1 (cổng khác: 10.123.219.1:8001)")
    ap.add_argument("--seconds", type=float, default=120, help="thời gian đo (mặc định 120 s)")
    ap.add_argument("--label", default="maixcam")
    ap.add_argument("--compare", default=None, help="file report_*.json của bench_report.py để thêm cột PC")
    args = ap.parse_args(argv)
    ip = args.ip if ":" in args.ip else args.ip + ":8000"

    try:
        first = get(ip, "/bench")
    except Exception as e:
        raise SystemExit(f"Không kết nối được http://{ip}/bench ({e}).\n"
                         "  - MaixCAM đã cắm / cùng mạng chưa? Mở http://%s/ trên trình duyệt thử." % ip)
    if "mem_used_peak_mb" not in first:
        raise SystemExit("MaixCAM đang chạy code CŨ (chưa có /bench). Run Project lại maixcam_app bằng MaixVision.")

    get(ip, "/bench/reset")
    print(f"Đang đo trên MaixCAM {ip} trong {args.seconds:.0f} s (đứng / đi lại trước camera)...")
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        time.sleep(min(10, max(0.1, t_end - time.time())))
        s = get(ip, "/bench")
        print(f"  {s.get('frames', 0)} khung | pose {s.get('pose_ms_mean', 0):.1f} ms | {s.get('fps') or 0:.1f} FPS"
              f" | có người {100 * (s.get('person_ratio') or 0):.0f}% | RAM hệ thống peak {s.get('mem_used_peak_mb')} MB")
    s = get(ip, "/bench")
    if not s.get("frames"):
        raise SystemExit("Không đo được khung nào.")
    if s.get("clients"):
        print(f"[CHÚ Ý] Có {s['clients']} client đang xem luồng: thời gian có cả nén JPEG (encode_ms). "
              "Tắt web/trình duyệt rồi đo lại nếu muốn số của riêng pipeline.")
    if (s.get("person_ratio") or 0) < 0.5:
        print(f"[CHÚ Ý] Chỉ {100 * (s.get('person_ratio') or 0):.0f}% khung có người -> pose có thể nhanh hơn thực tế.")

    params = pt_params(s.get("model_file"))
    pc = None
    if args.compare:
        files = sorted(glob.glob(args.compare))
        if files:
            pc = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
            print("So sánh với:", files[-1])

    inp = s.get("input")
    # (tên, giá trị MaixCAM, giá trị PC)
    rows = [
        (f"Pose: {Path(s.get('model_file') or s.get('model') or 'model').stem}", None, None),
        ("Parameters", fmt(params), fmt(pc and pc["pose"]["params"])),
        ("Model size", fmt(s.get("model_file_mb"), " MB (INT8 cvimodel)"),
         fmt(pc and pc["pose"]["size_mb"], " MB (FP32)")),
        ("Input size", f"{inp[0]}x{inp[1]}" if inp else "—",
         "x".join(map(str, pc["pose"]["imgsz"][::-1])) if pc and pc["pose"].get("imgsz") else "—"),
        ("Latency", fmt(s.get("pose_ms_mean"), " ms/frame") + f" (p95 {fmt(s.get('pose_ms_p95'))})",
         fmt(pc and pc["pose"]["latency_ms"], " ms/frame")),
        ("Throughput", fmt(1000 / s["pose_ms_mean"] if s.get("pose_ms_mean") else None, " FPS", 1),
         fmt(pc and pc["pose"]["throughput"], " FPS", 1)),
    ]
    on_board = bool(s.get("fall_model"))
    total_params = (params + s["fall_params"]) if params and s.get("fall_params") else None
    if on_board:
        rows += [
            (f"Classifier: Shift-GCN ({'NPU, ' + Path(s['fall_model']).suffix[1:] if s.get('fall_backend') == 'npu' else 'numpy, CPU'})",
             None, None),
            ("Parameters", fmt(s.get("fall_params")), fmt(pc and pc["classifier"]["params"])),
            ("Model size", fmt(s.get("fall_model_kb"), " KB (" + Path(s["fall_model"]).suffix[1:] + ")", 1),
             fmt(pc and pc["classifier"]["size_mb"] * 1024, " KB (FP32)", 1)),
            ("Latency", fmt(s.get("model_ms_mean"), " ms/clip") + f" (p95 {fmt(s.get('model_ms_p95'))})",
             fmt(pc and pc["classifier"]["latency_ms"], " ms/clip")),
            ("Throughput", fmt(1000 / s["model_ms_mean"] if s.get("model_ms_mean") else None, " clips/sec", 0),
             fmt(pc and pc["classifier"]["throughput"], " clips/sec", 0)),
            ("Tần số lấy mẫu (train: 10/s)", fmt(s.get("sample_hz"), " lần/s", 2)
             + (" ⚠" if (s.get("sample_hz") or 10) < 9 else ""), "10.00 lần/s"),
            ("Lọc + cửa sổ + model + hậu xử lý", fmt(s.get("fall_ms_mean"), " ms"), "—"),
            ("  SkeletonFilter", fmt(s.get("filter_ms_mean"), " ms"), "—"),
            ("  Cửa sổ + chuẩn hóa", fmt(s.get("prep_ms_mean"), " ms"), "—"),
            ("  Hậu xử lý", fmt(s.get("post_ms_mean"), " ms"), "—"),
        ]
    else:
        rows += [
            ("Classifier", None, None),
            ("Chạy ở", "PC (chưa đưa lên MaixCAM)", (pc and pc["classifier"]["name"]) or "—"),
            ("Latency", "—", fmt(pc and pc["classifier"]["latency_ms"], " ms/clip")),
        ]
    rows += [
        ("Combined Pipeline" if on_board else "Board pipeline (camera → pose)", None, None),
        ("Total Parameters", fmt(total_params / 1e6 if total_params else None, "M"),
         fmt(pc and pc["pipeline"]["params"] / 1e6, "M")),
        ("Camera read", fmt(s.get("read_ms_mean"), " ms"), "—"),
    ]
    if on_board:
        rows.append(("Camera → quyết định (latency)",
                     fmt(s.get("e2e_ms_mean"), " ms") + f" (p95 {fmt(s.get('e2e_ms_p95'))})",
                     fmt(pc and pc["pipeline"]["latency_ms"], " ms/frame")))
    rows += [
        ("JPEG encode (gửi PC)", fmt(s.get("encode_ms_mean"), " ms"), "—"),
        ("Display", fmt(s.get("display_ms_mean"), " ms"), "—"),
        ("Vòng lặp board (loop)", fmt(s.get("loop_ms_mean"), " ms/frame") + f" (p95 {fmt(s.get('loop_ms_p95'))})",
         "—" if on_board else fmt(pc and pc["pipeline"]["latency_ms"], " ms/frame")),
        ("End-to-end throughput", fmt(s.get("fps"), " FPS", 1), fmt(pc and pc["pipeline"]["throughput"], " FPS", 1)),
        ("Memory", None, None),
        ("Process peak RAM (VmHWM)", fmt(s.get("vm_hwm_mb"), " MB", 1), "—"),
        ("System RAM before model", fmt(s.get("mem_used_before_mb"), " MB", 1), "—"),
        ("System RAM peak", fmt(s.get("mem_used_peak_mb"), " MB", 1) + f" / {fmt(s.get('mem_total_mb'), ' MB', 0)}",
         "—"),
        ("Pipeline RAM (peak − before)", fmt(s.get("mem_pipeline_mb"), " MB", 1),
         fmt(pc and pc["pipeline"].get("peak_ram_mb"), " MB", 1)),
    ]
    two = pc is not None

    w0 = max(len(r[0]) for r in rows) + 2
    w1 = max(len(r[1] or "") for r in rows) + 2
    w2 = max(len(r[2] or "") for r in rows) + 2 if two else 0
    line = "=" * (w0 + w1 + w2 + 2)
    print("\n" + line)
    print(f"{'Metric':{w0}s}{'MaixCAM':>{w1}s}" + (f"{'PC':>{w2}s}" if two else ""))
    print("-" * len(line))
    for name, a, b in rows:
        if a is None:
            print(name)
        else:
            print(f"  {name:{w0 - 2}s}{a:>{w1}s}" + (f"{b:>{w2}s}" if two else ""))
    print(line)
    print(f"{s['frames']} khung trong {s.get('since_s')} s | có người {100 * (s.get('person_ratio') or 0):.0f}%"
          + (f" | PC: {pc['device']}" if two else ""))

    head = "| Metric | MaixCAM |" + (" PC |" if two else "")
    md = [head, "|---|---:|" + ("---:|" if two else "")]
    for name, a, b in rows:
        md.append(f"| **{name}** | |" + (" |" if two else "") if a is None
                  else f"| {name} | {a} |" + (f" {b} |" if two else ""))
    md.append(f"\n*MaixCAM: {s['frames']} khung trong {s.get('since_s')} s, {100 * (s.get('person_ratio') or 0):.0f}% "
              f"khung có người.*")

    def tex(x):
        return x.replace("_", r"\_").replace("%", r"\%").replace("→", r"$\rightarrow$").replace("−", "-")
    ncol = 3 if two else 2
    tx = [r"\begin{tabular}{l" + "r" * (ncol - 1) + "}", r"\toprule",
          r"\textbf{Metric} & \textbf{MaixCAM}" + (r" & \textbf{PC}" if two else "") + r" \\", r"\midrule"]
    first = True
    for name, a, b in rows:
        if a is None:
            if not first:
                tx.append(r"\midrule")
            tx.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{tex(name)}}}}} \\")
            first = False
        else:
            tx.append(f"{tex(name)} & {tex(a)}" + (f" & {tex(b)}" if two else "") + r" \\")
    tx += [r"\bottomrule", r"\end{tabular}"]

    out = resolve("output/bench")
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"maix_{args.label}_{datetime.now():%Y%m%d_%H%M%S}"
    stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    stem.with_suffix(".tex").write_text("\n".join(tx) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(dict(maixcam=s, pose_params=params, ip=ip,
                                                         seconds=args.seconds, compare=args.compare),
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu: {stem}.md / .tex / .json")


if __name__ == "__main__":
    main()
