"""Giao diện web (Flask) — lớp mỏng phía trên src/. Không chứa logic xử lý pose/lọc/ghi CSV riêng.

    python run_web.py        ->   http://127.0.0.1:5000

Chức năng (tương đương giao diện web của fall_app):
  - Xem trực tiếp (MJPEG) video / webcam / luồng RTSP kèm skeleton, FPS và latency.
  - Phát lại video đã trích bằng trình phát của trình duyệt (tạm dừng, tua được),
    khung xương vẽ chồng theo đúng thời điểm, lấy từ CSV đã trích.
  - Bật/tắt vẽ skeleton; bật/tắt SkeletonFilter khi xem trực tiếp.
  - Dải trạng thái: rảnh / đang xem trực tiếp / đang trích, FPS, tổng khung đã xử lý, số video, thời gian chạy.
  - Quét data/videos -> metadata.csv và báo lỗi nhãn / rò rỉ theo người.
  - Trích skeleton (CSV) cho một hoặc tất cả video, chạy nền, theo dõi tiến độ, hủy được.
  - Kiểm tra CSV có dùng được cho xử lý chuỗi không.
  - Tải CSV (dạng rộng gốc của CSVExporter, hoặc dạng dài) và bảng tóm tắt trích xuất.
  - Phát hiện ngã bằng FallClassifier (ST-GCN / LSTM...) nếu cấu hình classifier.checkpoint:
    nhãn NGÃ/không ngã trên luồng trực tiếp, đường P(ngã) theo thời gian trên trình phát.
  - Tạo dataset cửa sổ (windows.npz) từ NPZ đã trích + nhãn metadata, cho bước huấn luyện.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from pathlib import Path

import cv2
import pandas as pd
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from src.constants import SKELETON_EDGES

from . import csv_tools
from .config import resolve
from .engine import (Cancelled, OpenedSource, Stats, build_classifier, build_drawer, build_estimator,
                     build_filter, classifier_path, config_tag, csv_dir, draw_fall, draw_hud, estimate,
                     list_checkpoints,
                     extract_to_csv, load_npz, make_video_id, npz_dir, output_dir, pose_tag,
                     predict_video)
from .videos import load_metadata, scan_videos, validate

ID_RE = re.compile(r"^[0-9A-Za-z_]+$")


def create_app(cfg: dict, check_model: bool = True) -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    lock = threading.Lock()
    state = {
        "stream": None,          # dict(stop=Event, done=Event, src=OpenedSource|None) của luồng đang xem
        "live": {},              # số liệu luồng đang xem
        "jobs": {},              # job trích skeleton
        "total_frames": 0,       # tổng khung đã qua Pose từ lúc mở server (xem + trích)
        "last_frame_at": 0.0,
        "started": time.time(),
    }

    if check_model:
        # Tạo thử một lần để báo lỗi sớm (thiếu ultralytics / sai đường dẫn model).
        build_estimator(cfg).close()

    # Bộ phân loại ngã: lỗi (thiếu torch, sai checkpoint) chỉ báo trên giao diện, web vẫn chạy.
    clf_state = {"obj": None, "error": None, "name": None, "path": None}
    clf_lock = threading.Lock()
    cfg.setdefault("classifier", {})

    def _load_classifier() -> None:
        """(Nạp lại) bộ phân loại theo cfg["classifier"]["checkpoint"]. Luồng mới mở sau đó dùng model mới."""
        path = classifier_path(cfg)
        new = {"obj": None, "error": None, "name": None, "path": None}
        if path is None:
            ck = cfg["classifier"].get("checkpoint")
            new["error"] = ("chưa có checkpoint trong models/ (best_lstm.pt / best_stgcn.pt)"
                            if str(ck).lower() == "auto" else "chưa chọn model (classifier.checkpoint)")
        else:
            try:
                c = build_classifier({**cfg, "classifier": {**cfg["classifier"], "checkpoint": str(path)}})
                new.update(obj=c, name=c.name, path=_shown(path))
            except ModuleNotFoundError as e:
                new["error"] = f"chưa cài {e.name} (pip install torch)"
            except Exception as e:
                new["error"] = f"không nạp được {path.name}: {e}"
        with clf_lock:
            clf_state.update(new)

    out_dir = output_dir(cfg)
    cdir = csv_dir(cfg)
    meta_path = resolve(cfg["paths"]["metadata"])
    draw_conf = (cfg.get("draw") or {}).get("min_conf", 0.3)
    filter_default = bool((cfg.get("filter") or {}).get("enabled", True))

    def _meta() -> pd.DataFrame:
        return load_metadata(meta_path)

    def _csv(video_id: str) -> Path:
        return cdir / f"{video_id}.csv"

    def _shown(p: Path) -> str:
        root = resolve(".")
        return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)

    def _count_frame():
        with lock:
            state["total_frames"] += 1
            state["last_frame_at"] = time.time()

    _load_classifier()

    def _clf_info() -> dict:
        c = clf_state["obj"]
        return dict(name=clf_state["name"], error=clf_state["error"], path=clf_state["path"],
                    arch=c.arch if c else None, threshold=c.fall_threshold if c else None,
                    window_size=c.window_size if c else None, stride_frames=c.stride_frames if c else None,
                    params=c.num_params if c else None, available=list_checkpoints(cfg))

    # ------------------------------------------------------------------ trang chính
    @app.get("/")
    def index():
        p = cfg["pose"]
        return render_template("index.html", pose_tag=pose_tag(cfg), backend=p.get("backend", "yolo"),
                               conf=p.get("conf"), target_fps=cfg["source"].get("target_fps"),
                               filter_on=filter_default, out_dir=_shown(cdir), cfg_tag=config_tag(cfg),
                               edges=[list(e) for e in SKELETON_EDGES], draw_conf=draw_conf,
                               clf_name=clf_state["name"], clf_error=clf_state["error"],
                               clf_info=_clf_info())

    # ------------------------------------------------------------------ trình phát (file gốc, tua/tạm dừng được)
    def _video_path(video_id: str) -> Path:
        df = _meta()
        if not ID_RE.match(video_id) or video_id not in set(df.video_id):
            abort(404)
        p = resolve(df.loc[df.video_id == video_id, "path"].iloc[0]).resolve()
        if not p.is_file():
            abort(404)
        return p

    @app.get("/video/<video_id>")
    def video_file(video_id):
        return send_file(_video_path(video_id), conditional=True)   # HTTP Range -> tua được

    @app.get("/api/skeleton/<video_id>")
    def api_skeleton(video_id):
        if not ID_RE.match(video_id):
            abort(404)
        p = _csv(video_id)
        if not p.exists():
            return jsonify(error="chưa có skeleton"), 404
        return jsonify(csv_tools.to_json_payload(p))

    @app.get("/api/videos")
    def api_videos():
        rows = []
        for r in _meta().itertuples():
            rows.append(dict(video_id=r.video_id, label=r.label, path=r.path,
                             duration_s=None if pd.isna(r.duration_s) else float(r.duration_s),
                             split="" if pd.isna(r.split) else r.split,
                             missing="[MISSING FILE]" in str(r.note),
                             has_skeleton=_csv(r.video_id).exists(),
                             has_npz=(npz_dir(cfg) / f"{r.video_id}.npz").exists()))
        return jsonify(rows)

    @app.post("/api/scan")
    def api_scan():
        try:
            df, problems = scan_videos(resolve(cfg["paths"]["videos"]), meta_path)
        except Exception as e:
            return jsonify(error=str(e)), 400
        counts = df.groupby("label").size().to_dict() if len(df) else {}
        return jsonify(videos=len(df), counts=counts, problems=problems[:100])

    @app.get("/api/validate")
    def api_validate():
        return jsonify(problems=validate(_meta())[:100])

    # ------------------------------------------------------------------ xem trực tiếp (MJPEG)
    def _resolve_source(text: str):
        text = (text or "").strip()
        df = _meta()
        if text in set(df.video_id):
            return str(resolve(df.loc[df.video_id == text, "path"].iloc[0])), text
        return text, None

    def _stop_stream(wait: float = 0.0):
        cur = state["stream"]
        if cur is None:
            return
        cur["stop"].set()
        src = cur.get("src")
        if src is not None and hasattr(src.src, "stop"):
            src.src.stop()                         # StreamSource: thoát khỏi chờ khung ngay
        if wait:
            cur["done"].wait(wait)                 # đợi nhả webcam trước khi mở lại

    @app.get("/stream")
    def stream():
        spec, vid = _resolve_source(request.args.get("source", "0"))
        realtime = request.args.get("realtime", "1") == "1"
        show_skeleton = request.args.get("skeleton", "1") == "1"
        use_filter = request.args.get("filter", "1" if filter_default else "0") == "1"
        use_clf = request.args.get("classify", "1") == "1" and clf_state["obj"] is not None
        _stop_stream(wait=3.0)                    # chỉ một luồng xem tại một thời điểm
        me = {"stop": threading.Event(), "done": threading.Event(), "src": None}
        state["stream"] = me

        def gen():
            src = estimator = None
            try:
                try:
                    src = OpenedSource(spec, cfg, video_id=vid)
                    me["src"] = src
                    estimator = build_estimator(cfg)
                except Exception as e:
                    state["live"] = {"error": str(e)}
                    return
                flt = build_filter(cfg, enabled=use_filter)
                drawer = build_drawer(cfg)
                clf = None
                if use_clf:
                    try:
                        # mỗi luồng một bộ đệm cửa sổ riêng, cùng checkpoint đang chọn trên web
                        clf = build_classifier({**cfg, "classifier": {
                            **cfg["classifier"], "checkpoint": str(classifier_path(cfg))}})
                    except Exception as e:
                        state["live"] = {"error": f"bộ phân loại: {e}"}
                        return
                stats, wall0 = Stats(), time.perf_counter()
                hist: list[list[float]] = []          # [t, P(ngã)] mỗi lần dự đoán, cho biểu đồ
                events: list[dict] = []               # các lần chuyển sang NGÃ
                prev_label = 0
                state["live"] = dict(source=src.video_id, frames=0)
                for pkt in src.frames():
                    if me["stop"].is_set():
                        break
                    pose, lat = estimate(estimator, flt, pkt.frame, pkt.idx, pkt.t)
                    stats.add(pose, lat)
                    _count_frame()
                    if realtime and not src.is_live:          # phát file đúng tốc độ thật
                        delay = pkt.t - (time.perf_counter() - wall0)
                        if delay > 0:
                            time.sleep(delay)
                    # Pose vẫn chạy khi tắt skeleton, nên FPS và latency vẫn đúng
                    img = drawer.draw(pkt.frame, pose) if show_skeleton else pkt.frame.copy()
                    fall = {}
                    if clf is not None:
                        t0 = time.perf_counter()
                        res = clf.update(pose)                    # None nếu chưa tới lượt dự đoán
                        fall = dict(fall_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                                    threshold=clf.fall_threshold, window_size=clf.window_size,
                                    collected=len(clf.buffer))
                        if res is not None:                       # vừa có dự đoán mới
                            hist.append([round(pkt.t, 2), round(res["probability"], 4)])
                            del hist[:-300]
                            if res["label"] == 1 and prev_label == 0:
                                events.append(dict(t=round(pkt.t, 2), p=round(res["probability"], 3)))
                                del events[:-50]
                            prev_label = res["label"]
                        if clf.last is not None:
                            fall.update(fall_prob=round(clf.last["probability"], 3),
                                        fall_label=clf.last["label"])
                        fall.update(fall_hist=list(hist), fall_events=list(events))
                        draw_fall(img, clf.last)
                    s = stats.summary()
                    draw_hud(img, [f"{pose_tag(cfg)}{' +filter' if flt else ''}  t={pkt.t:.1f}s",
                                   f"pose {lat:.0f} ms | e2e {s['e2e_fps']:.1f} fps"])
                    state["live"] = dict(source=src.video_id, t=round(pkt.t, 2), valid=bool(pose.valid),
                                         latency_ms=round(lat, 1), filter=flt is not None,
                                         classifier=clf.name if clf else None,
                                         classifier_off=None if clf else (
                                             "đã tắt ô Phát hiện ngã" if clf_state["obj"] is not None
                                             else clf_state["error"]),
                                         out_fps=round(src.out_fps, 2), **fall, **s)
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            finally:
                if src is not None:
                    src.close()
                if estimator is not None:
                    estimator.close()
                if state["live"] and state["stream"] is me:
                    state["live"]["finished"] = True
                me["done"].set()

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/stop")
    def api_stop():
        _stop_stream()
        return jsonify(ok=True)

    @app.get("/api/live")
    def api_live():
        return jsonify(state["live"])

    # ------------------------------------------------------------------ trạng thái hệ thống
    @app.get("/api/status")
    def api_status():
        df = _meta()
        n_sk = sum(_csv(v).exists() for v in df.video_id) if len(df) else 0
        with lock:
            running = [j for j in state["jobs"].values() if j["status"] == "running"]
            total = state["total_frames"]
            last = state["last_frame_at"]
        live = state["live"]
        # Dựa vào thời điểm khung cuối: đóng tab giữa chừng thì generator dừng mà không kịp báo.
        fresh = time.time() - last < 2.0
        streaming = bool(live) and fresh and not live.get("finished") and not live.get("error") \
            and live.get("frames", 0) > 0
        if running:
            j = running[0]
            prog = f" ({j['cur_frames']}/{j['cur_total'] or '?'} khung)" if j.get("current") else ""
            status, detail = "đang trích skeleton", f"{j.get('current') or ''}{prog}"
        elif streaming:
            status, detail = "đang xem trực tiếp", live.get("source", "")
        else:
            status, detail = "rảnh", ""
        return jsonify(status=status, detail=detail, pose=pose_tag(cfg),
                       target_fps=cfg["source"].get("target_fps"), cfg_tag=config_tag(cfg),
                       videos=len(df), skeletons=n_sk, total_frames=total,
                       classifier=clf_state["name"], classifier_error=clf_state["error"],
                       fall_label=live.get("fall_label") if streaming else None,
                       fall_prob=live.get("fall_prob") if streaming else None,
                       uptime_s=int(time.time() - state["started"]),
                       live_fps=live.get("e2e_fps") if streaming else None,
                       live_frames=live.get("frames") if streaming else None)

    # ------------------------------------------------------------------ kiểm tra CSV cho xử lý chuỗi
    @app.get("/api/check/<video_id>")
    def api_check(video_id):
        if not ID_RE.match(video_id):
            abort(404)
        p = _csv(video_id)
        if not p.exists():
            return jsonify(error="chưa có skeleton"), 404
        return jsonify(csv_tools.check_csv(p, cfg.get("check")).to_dict())

    # ------------------------------------------------------------------ phân loại ngã offline + dataset
    @app.get("/api/predict/<video_id>")
    def api_predict(video_id):
        if not ID_RE.match(video_id):
            abort(404)
        if clf_state["obj"] is None:
            return jsonify(error=f"bộ phân loại: {clf_state['error']}"), 400
        p = npz_dir(cfg) / f"{video_id}.npz"
        if not p.exists():
            return jsonify(error="chưa có NPZ (trích lại skeleton cho video này)"), 404
        with clf_lock:                                # một model dùng chung, không gọi song song
            return jsonify(predict_video(clf_state["obj"], load_npz(p)))

    @app.get("/api/classifier")
    def api_classifier():
        return jsonify(_clf_info())

    @app.post("/api/classifier")
    def api_set_classifier():
        """Đổi model phân loại: {"checkpoint": "models/best_lstm.pt" | "auto" | null}."""
        ck = (request.get_json(silent=True) or {}).get("checkpoint")
        if ck not in (None, "", "auto") and ck not in list_checkpoints(cfg):
            return jsonify(error=f"không có checkpoint {ck} trong models/"), 400
        cfg["classifier"]["checkpoint"] = ck or None
        _load_classifier()
        _stop_stream()                       # luồng đang xem dùng model cũ -> trang sẽ mở lại
        info = _clf_info()
        return jsonify(info), (200 if info["name"] or not ck else 400)

    # ------------------------------------------------------------------ so sánh model (trang /compare)
    compare_state = {"last": None}
    compare_lock = threading.Lock()

    def _compare_options() -> dict:
        import importlib.util
        from .compare import list_datasets
        cks = list_checkpoints(cfg)
        pref = [c for c in ("models/best_stgcn.pt", "models/best_lstm.pt") if c in cks]
        default = pref if len(pref) >= 2 else cks[:2]
        has_torch = importlib.util.find_spec("torch") is not None
        return dict(checkpoints=cks, datasets=list_datasets(), default_models=default,
                    auto_run=has_torch and len(default) >= 2,
                    error=None if has_torch else "Chưa cài PyTorch (pip install torch) nên chưa chạy được model.")

    @app.get("/compare")
    def compare_page():
        from .compare import render_report
        return render_report(compare_state["last"], standalone=False, options=_compare_options())

    @app.post("/api/compare")
    def api_compare():
        from .compare import compare_models, list_datasets
        body = request.get_json(silent=True) or {}
        cks = [c for c in body.get("checkpoints") or [] if c in list_checkpoints(cfg)]
        data = body.get("data") or "data/dataset.npz"
        if data not in list_datasets():
            return jsonify(error=f"không có dataset {data}"), 400
        if not cks:
            return jsonify(error="chọn ít nhất một checkpoint trong models/"), 400
        try:
            with compare_lock:
                res = compare_models(cks, data, split=body.get("split", "test"),
                                     device=(cfg.get("classifier") or {}).get("device") or "cpu",
                                     runs=int(body.get("runs", 100)))
                compare_state["last"] = res
        except ModuleNotFoundError as e:
            return jsonify(error=f"chưa cài {e.name} (pip install torch)"), 400
        except Exception as e:
            return jsonify(error=str(e)), 400
        return jsonify(res)

    @app.get("/compare/export.<fmt>")
    def compare_export(fmt):
        import io
        import json
        from .compare import render_report, to_csv_rows
        res = compare_state["last"]
        if res is None:
            return jsonify(error="chưa chạy so sánh"), 404
        names = "_vs_".join(m["name"] for m in res["models"])
        if fmt == "json":
            buf = io.BytesIO(json.dumps(res, ensure_ascii=False, indent=1).encode("utf-8"))
            return send_file(buf, as_attachment=True, download_name=f"compare_{names}.json", mimetype="application/json")
        if fmt == "csv":
            rows = to_csv_rows(res)
            buf = io.BytesIO(pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"))
            return send_file(buf, as_attachment=True, download_name=f"compare_{names}.csv", mimetype="text/csv")
        if fmt == "html":
            buf = io.BytesIO(render_report(res, standalone=True).encode("utf-8"))
            return send_file(buf, as_attachment=True, download_name=f"compare_{names}.html", mimetype="text/html")
        abort(404)

    @app.post("/api/build-dataset")
    def api_build_dataset():
        from .dataset import build_dataset
        try:
            path, stats, notes = build_dataset(cfg)
        except Exception as e:
            return jsonify(error=str(e)), 400
        return jsonify(path=_shown(path), notes=notes[:100], **stats)

    # ------------------------------------------------------------------ trích skeleton (chạy nền)
    def _save_summary(rows: list[dict]):
        if not rows:
            return
        path = out_dir / "_extract_summary.csv"
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        new = pd.DataFrame(rows)
        if len(old) and "video_id" in old:
            old = old[~old.video_id.isin(new.video_id)]
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)

    def _run_job(job_id, items, overwrite):
        job = state["jobs"][job_id]
        summary = []
        estimator = None
        try:
            estimator = build_estimator(cfg)
        except Exception as e:
            with lock:
                job["log"].append({"video_id": "-", "result": f"LỖI tạo model: {e}"})
                job["status"] = "finished"
            return

        def progress(n, total):
            with lock:
                state["total_frames"] += max(0, n - job["cur_frames"])
                job["cur_frames"], job["cur_total"] = n, total
                state["last_frame_at"] = time.time()

        for spec, vid in items:
            if job["cancel"]:
                break
            with lock:
                job["current"], job["cur_frames"], job["cur_total"] = vid, 0, None
            out = _csv(vid)
            # CSV trích trước khi có NPZ (bản cũ) thì trích lại để có NPZ cho dataset.
            if out.exists() and (npz_dir(cfg) / f"{vid}.npz").exists() and not overwrite:
                msg = "đã có, bỏ qua"
            else:
                try:
                    msg = extract_to_csv(spec, vid, out, cfg, estimator=estimator, progress=progress,
                                         should_stop=lambda: job["cancel"])
                    summary.append(dict(video_id=vid, **msg))
                except Cancelled:
                    msg = "đã hủy"
                except Exception as e:
                    msg = f"LỖI: {e}"
                    summary.append(dict(video_id=vid, error=str(e)))
            with lock:
                job["done"] += 1
                job["log"].append({"video_id": vid, "result": msg})
        estimator.close()
        _save_summary(summary)
        with lock:
            job["status"] = "cancelled" if job["cancel"] else "finished"
            job["current"] = None

    @app.post("/api/extract")
    def api_extract():
        body = request.get_json(silent=True) or {}
        target = (body.get("source") or "").strip()
        overwrite = bool(body.get("overwrite"))
        df = _meta()
        df = df[~df.note.astype(str).str.contains(r"\[MISSING FILE\]", regex=True)]
        if target in ("", "*", "all"):
            items = [(str(resolve(r.path)), r.video_id) for r in df.itertuples()]
        elif target in set(df.video_id):
            r = df[df.video_id == target].iloc[0]
            items = [(str(resolve(r.path)), r.video_id)]
        else:
            p = Path(target)
            if not p.is_absolute() and not p.exists():
                p = resolve(p)
            if not p.is_file():
                return jsonify(error=f"Không tìm thấy file video: {target} "
                                     "(trích skeleton chỉ dùng cho video file, không cho webcam/luồng)"), 400
            items = [(str(p), make_video_id(p.stem))]
        if not items:
            return jsonify(error="Không có video nào. Chép video vào data/videos/fall, data/videos/adl rồi bấm Quét video."), 400
        job_id = uuid.uuid4().hex[:8]
        with lock:
            state["jobs"][job_id] = dict(id=job_id, status="running", total=len(items), done=0, current=None,
                                         cur_frames=0, cur_total=None, log=[], cancel=False,
                                         started=time.strftime("%H:%M:%S"))
        threading.Thread(target=_run_job, args=(job_id, items, overwrite), daemon=True).start()
        return jsonify(job_id=job_id, total=len(items))

    @app.post("/api/jobs/<job_id>/cancel")
    def api_cancel(job_id):
        with lock:
            job = state["jobs"].get(job_id)
            if job is None:
                abort(404)
            job["cancel"] = True
        return jsonify(ok=True)

    @app.get("/api/jobs")
    def api_jobs():
        with lock:
            return jsonify([{k: v for k, v in j.items()} for j in list(state["jobs"].values())[-10:]])

    # ------------------------------------------------------------------ tải về
    def _existing_csv(video_id: str) -> Path:
        if not ID_RE.match(video_id):
            abort(404)
        p = _csv(video_id)
        if not p.exists():
            abort(404)
        return p

    @app.get("/download/<video_id>.csv")
    def dl_csv(video_id):
        fmt = request.args.get("format", "wide")
        p = _existing_csv(video_id)
        if fmt == "wide":
            return send_file(p, as_attachment=True, download_name=f"{video_id}.csv")
        if fmt == "long":
            out = csv_tools.to_long(p, out_dir / "long" / f"{video_id}.long.csv")
            return send_file(out, as_attachment=True, download_name=f"{video_id}.long.csv")
        abort(400)

    @app.get("/download/_extract_summary.csv")
    def dl_summary():
        p = out_dir / "_extract_summary.csv"
        if not p.exists():
            abort(404)
        return send_file(p, as_attachment=True)

    return app
