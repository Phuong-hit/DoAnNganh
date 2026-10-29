"""
make_intervals.py
-----------------
Đọc bảng chú thích của dataset (cột "File Name" và "Classes", ví dụ
"Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]") và ghi khoảng ngã vào
data/fall_intervals.csv (cột video_id,start_sec,end_sec) cho build_dataset.py.

video_id = "<subject>_<kind>_<tên file không đuôi>", trùng với extract_skeleton.py.

Chạy (mỗi lần một bảng = một subject, một loại):
    python make_intervals.py annotations.csv --subject "Subject 1" --kind Fall
Bảng .xlsx cũng được (cần pandas + openpyxl); hoặc dùng Excel: Save As -> CSV.
Chạy lại cho cùng video_id sẽ ghi đè dòng cũ, không bị trùng.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

FALL_RE = re.compile(r"fall\w*[^\[;]*\[\s*([\d.]+)\s*(?:to|-)\s*([\d.]+)\s*\]", re.IGNORECASE)


def parse_falls(text):
    """'Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]' -> [(3.4, 6.0)]."""
    return [(float(a), float(b)) for a, b in FALL_RE.findall(str(text or ""))]


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(path)
        return df.columns.tolist(), df.astype(str).to_dict("records")
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1258", "cp1252"):       # Excel hay lưu ANSI
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = text.splitlines()
    header = lines[0] if lines else ""
    delim = ";" if header.count(";") > header.count(",") else ","   # Excel VN hay dùng ";"
    r = csv.DictReader(lines, delimiter=delim)
    return r.fieldnames or [], list(r)


def find_col(cols, prefix):
    for c in cols:
        if str(c).strip().lower().startswith(prefix):
            return c
    sys.exit(f'Không thấy cột bắt đầu bằng "{prefix}" trong: {list(cols)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", help="bảng chú thích (.csv hoặc .xlsx)")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--kind", default="Fall")
    ap.add_argument("--out", default="data/fall_intervals.csv")
    args = ap.parse_args()
    if "_" in args.subject or "_" in args.kind:
        sys.exit('subject và kind không được chứa dấu "_".')

    cols, rows = read_rows(args.table)
    c_file, c_cls = find_col(cols, "file"), find_col(cols, "class")

    new = {}
    for r in rows:
        name = str(r[c_file]).strip()
        if not name or name.lower() == "nan":
            continue
        falls = parse_falls(r[c_cls])
        vid = f"{args.subject}_{args.kind}_{Path(name).stem}"
        if falls:
            new[vid] = falls
        else:
            print(f"[bỏ qua] {vid}: không thấy khoảng 'Falling' trong: {r[c_cls]!r}")

    out = Path(args.out)
    kept = []
    if out.exists():
        with open(out, newline="", encoding="utf-8-sig") as f:
            kept = [r for r in csv.DictReader(f) if r["video_id"] not in new]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "start_sec", "end_sec"])
        for r in kept:
            w.writerow([r["video_id"], r["start_sec"], r["end_sec"]])
        for vid, falls in new.items():
            for a, b in falls:
                w.writerow([vid, a, b])
                print(f"{vid}: ngã {a} -> {b} s")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()