#!/usr/bin/env python3
"""Điểm chạy giao diện web. Chạy từ thư mục gốc project (cùng cấp với src/).

  python run_web.py                         # web tại http://127.0.0.1:5000
  python run_web.py web --host 0.0.0.0      # cho máy khác trong mạng LAN truy cập
  python run_web.py scan                    # quét data/videos -> data/metadata.csv (giống nút "Quét video")
  python run_web.py check-csv               # kiểm tra mọi CSV của cấu hình hiện tại (thoát mã 1 nếu có lỗi)
  python run_web.py build-dataset           # NPZ đã trích + nhãn metadata -> output/<cấu hình>/dataset/windows.npz

Tùy chọn chung: --config configs/web.yaml   --set pose.backend=dummy --set source.target_fps=15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép "import src..." / "import web..." dù chạy từ thư mục khác.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.config import load_config, resolve  # noqa: E402


def cmd_web(cfg, args):
    from web.server import create_app
    try:
        app = create_app(cfg)
    except Exception as e:
        print(f"Không khởi động được: {e}\n"
              "Gợi ý: cài ultralytics (pip install ultralytics), kiểm tra pose.model_path, "
              "hoặc chạy thử không cần model: python run_web.py --set pose.backend=dummy")
        return 1
    print(f"Mở trình duyệt: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


def cmd_scan(cfg, args):
    from web.videos import scan_videos
    df, problems = scan_videos(resolve(cfg["paths"]["videos"]), resolve(cfg["paths"]["metadata"]))
    print(f"Đã ghi {resolve(cfg['paths']['metadata'])}: {len(df)} video")
    if len(df):
        print(df.groupby("label").size().to_string())
    if problems:
        print(f"\n{len(problems)} vấn đề cần sửa tay trong metadata.csv:")
        for p in problems[:50]:
            print("  -", p)
    else:
        print("Metadata hợp lệ.")


def cmd_check(cfg, args):
    from web.csv_tools import check_csv
    from web.engine import csv_dir
    p = Path(args.path) if args.path else csv_dir(cfg)
    files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    if not files:
        print(f"Không có file CSV nào trong {p}")
        return 1
    bad = 0
    for f in files:
        rep = check_csv(f, cfg.get("check"))
        bad += not rep.ok
        print(rep.to_text())
    print(f"\n{len(files) - bad}/{len(files)} file đạt")
    return 1 if bad else 0


def cmd_dataset(cfg, args):
    from web.dataset import build_dataset
    try:
        path, stats, notes = build_dataset(cfg, args.out)
    except ValueError as e:
        print(f"Lỗi: {e}")
        return 1
    print(f"Đã ghi {path}: {stats['windows']} cửa sổ {stats['window_size']} khung @ {stats['fps']} fps "
          f"(ngã {stats['fall']}, không ngã {stats['no_fall']}) từ {stats['videos']} video")
    for k, v in stats["by_split"].items():
        print(f"  split {k}: {v['windows']} cửa sổ, ngã {v['fall']}")
    for n in notes[:50]:
        print("  -", n)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Giao diện web phát hiện ngã",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config", default=None, help="file YAML (mặc định configs/web.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="ghi đè cấu hình")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("web", help="giao diện web (mặc định)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    sub.add_parser("scan", help="quét data/videos -> metadata.csv")
    p = sub.add_parser("check-csv", help="kiểm tra CSV cho xử lý chuỗi")
    p.add_argument("path", nargs="?", help="file hoặc thư mục CSV (mặc định: thư mục của cấu hình hiện tại)")
    p = sub.add_parser("build-dataset", help="tạo windows.npz cho huấn luyện (ST-GCN / LSTM)")
    p.add_argument("--out", default=None, help="file ra (mặc định output/<cấu hình>/dataset/windows.npz)")
    args = ap.parse_args(argv)
    if args.cmd is None:
        args = ap.parse_args((argv or sys.argv[1:]) + ["web"])
    cfg = load_config(args.config, args.set)
    fn = {"web": cmd_web, "scan": cmd_scan, "check-csv": cmd_check, "build-dataset": cmd_dataset}[args.cmd]
    return fn(cfg, args) or 0


if __name__ == "__main__":
    sys.exit(main())
