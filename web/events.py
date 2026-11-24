"""Lưu lại khoảnh khắc ngã để xem lại: N giây TRƯỚC cảnh báo + M giây SAU, kèm khung xương và P(ngã).

    EventRecorder  -- nhận từng khung từ Monitor, giữ bộ đệm vòng ~pre_s giây. Khi có cảnh báo: chốt
                      bộ đệm, gom thêm post_s giây (cảnh báo mới trong lúc gom -> kéo dài clip), rồi giao
                      cho một luồng riêng ghi ra đĩa (vòng lặp phát hiện không bao giờ chờ ổ đĩa).
    EventStore     -- thư mục event trên đĩa: liệt kê, đọc, gắn nhãn đúng/sai, lưu vĩnh viễn, xóa,
                      chỉ giữ max_events event gần nhất.

Một event (output/fall_events/<id>/):
    meta.json     thời điểm, nguồn, model, ngưỡng, cửa sổ model dùng lúc báo, nhãn người xem
    track.json    mỗi khung: t, valid, 17 khớp [x, y, conf] (chuẩn hóa 0-1, null = thiếu), bbox,
                  P(ngã) gần nhất, nhãn, gated, pred (khung này model vừa dự đoán)
    frames/000000.jpg ...   ảnh SẠCH (chưa vẽ), trình duyệt tự vẽ khung xương
    thumb.jpg     ảnh lúc cảnh báo
Ghi vào thư mục .tmp_<id> rồi mới đổi tên: web không bao giờ đọc phải event ghi dở.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

EVENT_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}(_[0-9]+)?$")
LABELS = (None, "fall", "false_alarm")        # chưa xem | báo đúng | báo sai


def _joints_list(joints) -> list | None:
    out = []
    for x, y, c in np.asarray(joints, dtype=float):
        out.append(None if not (math.isfinite(x) and math.isfinite(y)) or c <= 0
                   else [round(x, 4), round(y, 4), round(c, 3)])
    return out


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


# ============================================================================ ghi
class EventRecorder:
    def __init__(self, store: "EventStore", pre_s: float = 6.0, post_s: float = 5.0, max_clip_s: float = 30.0,
                 cooldown_s: float = 20.0, frame_width: int = 480, jpeg_quality: int = 75,
                 info: dict | None = None, on_saved=None):
        self.store = store
        self.pre_s, self.post_s, self.max_clip_s, self.cooldown_s = pre_s, post_s, max_clip_s, cooldown_s
        self.frame_width, self.jpeg_quality = int(frame_width), int(jpeg_quality)
        self.info = dict(info or {})            # source, model, threshold, window_size... ghi vào meta.json
        self.on_saved = on_saved
        self._buf: deque = deque()
        self._cap: dict | None = None           # event đang gom
        self._last_end = -1e9                    # t kết thúc event trước (cho cooldown)
        self._threads: list[threading.Thread] = []
        self.skipped = 0                         # cảnh báo bị bỏ qua vì đang trong cooldown

    @property
    def recording(self) -> dict | None:
        c = self._cap
        return None if c is None else dict(id=c["id"], alerts=len(c["alerts"]))

    def _encode(self, frame) -> bytes | None:
        h, w = frame.shape[:2]
        if w > self.frame_width:
            frame = cv2.resize(frame, (self.frame_width, round(h * self.frame_width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buf.tobytes() if ok else None

    def push(self, t: float, frame, pose, pred: dict | None, new_pred: bool, alert: bool) -> str | None:
        """Một khung đã xử lý. pred = kết quả model gần nhất (clf.last), new_pred = khung này vừa dự đoán,
        alert = khung này chuyển sang NGÃ. Trả id event nếu khung này mở một event mới."""
        rec = dict(t=float(t), jpeg=self._encode(frame), valid=bool(pose.valid),
                   joints=_joints_list(pose.joints) if pose.valid else None,
                   bbox=[round(float(v), 4) for v in pose.bbox] if pose.valid and pose.bbox is not None else None,
                   w=int(pose.width or frame.shape[1]), h=int(pose.height or frame.shape[0]),
                   p=None if pred is None else round(float(pred["probability"]), 4),
                   label=None if pred is None else int(pred["label"]),
                   gated=bool(pred.get("gated", False)) if pred else False,
                   pred=bool(new_pred))
        started = None
        if self._cap is not None:
            c = self._cap
            c["frames"].append(rec)
            if alert:
                c["alerts"].append(len(c["frames"]) - 1)
                c["t_end"] = min(max(c["t_end"], t + self.post_s), c["frames"][0]["t"] + self.max_clip_s)
            if t >= c["t_end"]:
                self._finish()
        else:
            self._buf.append(rec)
            while self._buf and t - self._buf[0]["t"] > self.pre_s:
                self._buf.popleft()
            if alert:
                if t - self._last_end < self.cooldown_s:
                    self.skipped += 1
                else:
                    frames = list(self._buf)
                    self._buf.clear()
                    started = self.store.new_id()
                    self._cap = dict(id=started, frames=frames, alerts=[len(frames) - 1],
                                     t_end=t + self.post_s, created=datetime.now().isoformat(timespec="seconds"),
                                     info=dict(self.info))
        return started

    def flush(self):
        """Nguồn dừng giữa chừng: ghi luôn phần đã gom (clip ngắn hơn nhưng không mất)."""
        if self._cap is not None:
            self._finish()

    def _finish(self):
        c, self._cap = self._cap, None
        self._last_end = c["frames"][-1]["t"]
        th = threading.Thread(target=self._save, args=(c,), daemon=True)
        th.start()
        self._threads = [x for x in self._threads if x.is_alive()] + [th]

    def _save(self, c):
        try:
            self.store.save(c)
            if self.on_saved:
                self.on_saved(c["id"])
        except Exception as e:                                       # ổ đầy / mất quyền ghi: không làm sập giám sát
            print(f"[events] không lưu được event {c['id']}: {e}")

    def wait_saved(self, timeout: float = 10.0):
        for th in list(self._threads):
            th.join(timeout)


# ============================================================================ kho trên đĩa
class EventStore:
    def __init__(self, root: Path, saved_root: Path, max_events: int = 5):
        self.root = Path(root)
        self.saved_root = Path(saved_root)
        self.max_events = int(max_events)
        self._lock = threading.Lock()
        self.version = 0                          # tăng mỗi khi danh sách đổi, web dùng để biết khi nào tải lại
        for d in (self.root, self.saved_root):
            d.mkdir(parents=True, exist_ok=True)
            for tmp in d.glob(".tmp_*"):          # event ghi dở lần trước (mất điện, tắt server)
                shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------ id / đường dẫn
    def new_id(self) -> str:
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        eid, n = base, 1
        while (self.root / eid).exists() or (self.saved_root / eid).exists() or eid == getattr(self, "_last_id", None):
            n += 1
            eid = f"{base}_{n}"
        self._last_id = eid
        return eid

    def path(self, eid: str) -> Path | None:
        if not EVENT_ID_RE.match(eid or ""):
            return None
        for d in (self.root, self.saved_root):
            if (d / eid / "meta.json").is_file():
                return d / eid
        return None

    # ------------------------------------------------------------------ ghi
    def save(self, c: dict) -> Path:
        frames = c["frames"]
        alerts = c["alerts"]
        a0 = alerts[0]
        t0 = frames[0]["t"]
        info = c.get("info") or {}
        W = int(info.get("window_size") or 0)
        tmp = self.root / f".tmp_{c['id']}"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "frames").mkdir(parents=True)
        for i, f in enumerate(frames):
            if f["jpeg"]:
                (tmp / "frames" / f"{i:06d}.jpg").write_bytes(f["jpeg"])
        if frames[a0]["jpeg"]:
            (tmp / "thumb.jpg").write_bytes(frames[a0]["jpeg"])
        track = [dict(t=round(f["t"] - t0, 3), valid=f["valid"], joints=f["joints"], bbox=f["bbox"],
                      p=f["p"], label=f["label"], gated=f["gated"], pred=f["pred"], img=bool(f["jpeg"]))
                 for f in frames]
        probs = [f["p"] for f in frames if f["p"] is not None]
        dur = frames[-1]["t"] - t0
        meta = dict(id=c["id"], created=c["created"], n_frames=len(frames),
                    duration_s=round(dur, 2), fps=round((len(frames) - 1) / dur, 2) if dur > 0 else None,
                    t_alert=round(frames[a0]["t"] - t0, 3), alert_frame=a0, alert_frames=alerts,
                    t_alerts=[round(frames[i]["t"] - t0, 3) for i in alerts],
                    window=[max(0, a0 - W + 1), a0] if W else None,
                    max_prob=max(probs) if probs else None, p_alert=frames[a0]["p"],
                    width=frames[a0]["w"], height=frames[a0]["h"],
                    valid_ratio=round(sum(f["valid"] for f in frames) / len(frames), 3),
                    label=None, note="", labeled_at=None, **info)
        _write_json(tmp / "track.json", track)
        _write_json(tmp / "meta.json", meta)
        with self._lock:
            os.replace(tmp, self.root / c["id"])
            self._prune()
            self.version += 1
        return self.root / c["id"]

    def _prune(self):
        """Giữ max_events event (không tính event đã lưu vĩnh viễn). Xóa trước: event đã gắn nhãn cũ nhất,
        sau đó mới tới event chưa xem cũ nhất — để cảnh báo chưa ai xem không bị mất sớm."""
        items = [self._meta(d) for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        items = [m for m in items if m]
        extra = len(items) - self.max_events
        if extra <= 0:
            return
        items.sort(key=lambda m: (m.get("label") is None, m["id"]))      # đã gắn nhãn trước, cũ trước
        for m in items[:extra]:
            shutil.rmtree(self.root / m["id"], ignore_errors=True)

    # ------------------------------------------------------------------ đọc
    @staticmethod
    def _meta(d: Path) -> dict | None:
        try:
            return json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list(self) -> dict:
        def scan(root, kept):
            out = []
            for d in root.iterdir():
                if d.is_dir() and not d.name.startswith("."):
                    m = self._meta(d)
                    if m:
                        m["kept"] = kept
                        out.append(m)
            return sorted(out, key=lambda m: m["id"], reverse=True)
        with self._lock:
            return dict(events=scan(self.root, False), saved=scan(self.saved_root, True),
                        max_events=self.max_events, version=self.version)

    def get(self, eid: str) -> dict | None:
        d = self.path(eid)
        if d is None:
            return None
        m = self._meta(d)
        try:
            track = json.loads((d / "track.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        m["kept"] = d.parent == self.saved_root
        return dict(meta=m, track=track)

    def frame_path(self, eid: str, n: int) -> Path | None:
        d = self.path(eid)
        p = None if d is None else d / "frames" / f"{int(n):06d}.jpg"
        return p if p is not None and p.is_file() else None

    def thumb_path(self, eid: str) -> Path | None:
        d = self.path(eid)
        p = None if d is None else d / "thumb.jpg"
        return p if p is not None and p.is_file() else None

    # ------------------------------------------------------------------ thao tác của người xem
    def set_label(self, eid: str, label: str | None, note: str | None = None) -> dict | None:
        if label not in LABELS:
            raise ValueError(f"nhãn không hợp lệ: {label}")
        with self._lock:
            d = self.path(eid)
            if d is None:
                return None
            m = self._meta(d)
            m["label"] = label
            m["labeled_at"] = datetime.now().isoformat(timespec="seconds") if label else None
            if note is not None:
                m["note"] = str(note)[:500]
            _write_json(d / "meta.json", m)
            self.version += 1
            return m

    def keep(self, eid: str, keep: bool = True) -> bool:
        """Lưu vĩnh viễn (chuyển sang saved_root, không bị xóa vòng) hoặc trả lại danh sách thường."""
        with self._lock:
            d = self.path(eid)
            if d is None:
                return False
            dst = (self.saved_root if keep else self.root) / eid
            if d != dst:
                os.replace(d, dst)
                if not keep:
                    self._prune()
            self.version += 1
            return True

    def delete(self, eid: str) -> bool:
        with self._lock:
            d = self.path(eid)
            if d is None:
                return False
            shutil.rmtree(d, ignore_errors=True)
            self.version += 1
            return True

    def as_item(self, eid: str) -> dict | None:
        """Chuỗi khung xương của event theo quy ước skeleton_io (seq, mask, t, fps, aspect) — để chạy lại
        bằng model khác (engine.predict_video) hoặc đưa vào dataset."""
        ev = self.get(eid)
        if ev is None:
            return None
        from src.pose_estimate import PoseResult, empty_joints
        from src.skeleton_io import from_poses
        m, poses = ev["meta"], []
        for i, f in enumerate(ev["track"]):
            j = empty_joints()
            for k, p in enumerate(f["joints"] or []):
                if p is not None:
                    j[k] = p
            poses.append(PoseResult(i, f["t"], bool(f["valid"]), j, m.get("width") or 0, m.get("height") or 0))
        return from_poses(poses)
