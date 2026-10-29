# Fall Web — giao diện web cho pipeline pose (newfilecode)

Phục vụ đồ án *Hệ thống phát hiện ngã dựa trên skeleton trên thiết bị biên*.

- **Xử lý** (nguồn video → ước lượng tư thế → lọc → ghi CSV / vẽ → phân loại ngã) nằm trong `src/`, là bản
  **code_1** (One Euro filter, ByteTrack, bbox + score, ST-GCN). Logic từng tệp xem ở [`HUONG_DAN.md`](HUONG_DAN.md).
- **Phát hiện ngã** qua `src/classifier.py` (`FallClassifier`), **dùng chung cho mọi kiến trúc**: ST-GCN có sẵn,
  LSTM (B1) chỉ cần thêm `src/model_lstm.py` — xem [mục 10](#10-b1-thêm-model-lstm).
- **Giao diện web** (`web/`) có các chức năng như giao diện web của fall_app, viết lại để gọi các lớp
  `FileSource`, `StreamSource`, `PoseEstimate`, `SkeletonFilter`, `SkeletonDraw`, `CSVExporter` của `src/`.

```
 Nguồn                       src/ (giữ nguyên)                                   Web
┌──────────────┐ khung ┌──────────────┐ PoseResult ┌────────────────┐ ┌──────────────────────────┐
│ FileSource   │──────►│ PoseEstimate │───────────►│ SkeletonFilter │►│ CSVExporter → CSV        │
│ StreamSource │       │ (YOLO-pose)  │            │ (tùy chọn)     │ │ + NPZ (skeleton_io)      │
└──────────────┘       └──────────────┘            └────────────────┘ │ SkeletonDraw → MJPEG     │
                                                           │          └──────────────────────────┘
                                                           ▼
                                      FallClassifier (cửa sổ 20 khung, ST-GCN / LSTM) → NGÃ / không ngã
```

## 1. Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # có ultralytics (kéo theo PyTorch)
python -m unittest discover tests    # kiểm thử bằng pose giả, không cần model
```

Trọng số `models/yolov8n-pose.pt` đã có sẵn (lấy từ fall_app).

## 2. Chạy

Luôn chạy từ thư mục gốc project (cùng cấp với `src/`).

```bash
python run_web.py                                  # mở http://127.0.0.1:5000
python run_web.py web --host 0.0.0.0 --port 5000   # cho máy khác trong mạng LAN mở
python run_web.py --set pose.backend=dummy         # chạy thử giao diện khi chưa cài ultralytics
python run_web.py scan                             # quét video -> data/metadata.csv (giống nút "Quét video")
python run_web.py check-csv                        # kiểm tra mọi CSV của cấu hình hiện tại
python run_web.py build-dataset                    # NPZ + nhãn -> output/<cấu hình>/dataset/windows.npz
python run_web.py --set classifier.checkpoint=models/stgcn_fall.pt   # bật phát hiện ngã
```

Quy trình thường dùng:

1. Chép video vào `data/videos/fall/` (video ngã) và `data/videos/adl/` (sinh hoạt thường).
2. Mở web, bấm **Quét video** → sinh `data/metadata.csv`. Mở file này điền tay `fall_start_s`, `fall_end_s`,
   `subject_id`, `split`… (các cột này được giữ nguyên khi quét lại).
3. Bấm **Trích tất cả** (hoặc *trích* ở từng dòng) → CSV ở `output/<cấu hình>/csv/<video_id>.csv`.
4. Bấm tên video để phát lại kèm khung xương; bấm *kiểm tra* để xem CSV có dùng được cho xử lý chuỗi không.
5. Bấm **Tạo dataset** (hoặc `run_web.py build-dataset`) → `windows.npz` để huấn luyện ST-GCN / LSTM.
6. Có checkpoint thì đặt `classifier.checkpoint` → luồng trực tiếp có nhãn **FALL / no fall**, trình phát có dải P(ngã).

Ghi đè cấu hình không cần sửa file: `python run_web.py --set source.target_fps=15 --set filter.enabled=false`.

## 3. Chức năng của giao diện web

| Chức năng | Mô tả |
|---|---|
| Xem trực tiếp | Nhập `video_id`, đường dẫn file, số webcam (`0`) hoặc URL (`rtsp://`, `http://`) rồi bấm **Xem**. Ảnh MJPEG có skeleton, latency pose và FPS đầu-cuối. File phát đúng tốc độ thật. Chỉ một luồng xem tại một thời điểm |
| Trình phát | `video_id` có trong metadata phát bằng trình phát của trình duyệt (tạm dừng, tua được). Khung xương vẽ chồng theo đúng thời điểm, lấy từ CSV đã trích |
| Hiện skeleton | Trình phát: chỉ ẩn lớp vẽ. Luồng trực tiếp: mở lại luồng với `skeleton=0`. Pose vẫn chạy nên số liệu vẫn đúng |
| Làm mượt (SkeletonFilter) | Bật/tắt bộ lọc khi xem trực tiếp, để so sánh có/không lọc bằng mắt. CSV trích ra theo `filter.enabled` trong cấu hình |
| Dải trạng thái | Rảnh / đang xem trực tiếp / đang trích skeleton (kèm số khung đã xử lý của video hiện tại), FPS, tổng khung đã xử lý, số video, số video đã có skeleton, thời gian chạy |
| Quét video | Quét `data/videos/{fall,adl}` → `metadata.csv`, báo video thiếu thời điểm ngã, nhãn sai, file mất, rò rỉ theo người giữa các tập |
| Trích skeleton | Một video, tất cả, hoặc một đường dẫn file. Chạy nền, có thanh tiến độ, **Hủy** được. Video đã có thì bỏ qua (tích *chạy lại* hoặc bấm *trích lại* để ghi đè). Ghi vào file tạm rồi mới đổi tên, nên video trích dở không bị coi là "đã có" |
| Kiểm tra CSV | Xem mục 5 |
| Trạng thái ngã | Khung lớn phía trên video: **NGÃ! / KHÔNG NGÃ / ĐANG GOM KHUNG / CHƯA BẬT** (kèm lý do), tỉ lệ ngã P(ngã) dạng % và thanh có vạch ngưỡng, biểu đồ P(ngã) theo thời gian, danh sách các lần phát hiện ngã. Ô **Model** đổi checkpoint ngay trên web. Mặc định `classifier.checkpoint: auto` (best_lstm.pt, không có thì best_stgcn.pt) |
| Phát hiện ngã | Khi có `classifier.checkpoint`: luồng trực tiếp vẽ nhãn **FALL p / no fall p** ở góc phải trên, thống kê có P(ngã) và thời gian phân loại. Trình phát vẽ nhãn và dải P(ngã) theo thời gian (tính từ NPZ, cùng cách cắt/chuẩn hóa). Tắt bằng ô *Phát hiện ngã*. Không có checkpoint hoặc thiếu torch: web vẫn chạy, đầu trang báo lý do |
| Tạo dataset | NPZ đã trích + `fall_start_s/fall_end_s/split/subject_id` trong metadata → `windows.npz` (mục 10) |
| Tải về | CSV gốc (dạng rộng, đúng định dạng `CSVExporter`), CSV dạng dài (1 dòng/khớp), bảng `_extract_summary.csv` (số khung, tỉ lệ có người, `missing_ratio`, latency, FPS từng video) |

## 4. Cấu trúc thư mục

```
fall_web/
├── run_web.py              # điểm chạy (web / scan / check-csv / build-dataset)
├── train.py  evaluate.py  main.py                       # B1: huấn luyện, so sánh, chạy video (HUONG_DAN_B1.md)
├── extract_skeleton.py  make_intervals.py  build_dataset.py   # tạo dataset kiểu code_1
├── HUONG_DAN_B1.md         # hướng dẫn B1 chi tiết
├── configs/web.yaml        # mọi thiết lập: FPS, model, ngưỡng, bộ lọc, tham số kiểm tra
├── configs/code1.yaml      # như trên nhưng tham số pose giống lúc tạo data/dataset.npz
├── src/                    # newfilecode — GIỮ NGUYÊN, xem HUONG_DAN.md
│   ├── constants.py  pose_estimate.py  skeleton_filter.py  skeleton_draw.py
│   ├── video_process.py  webcam.py  csv_export.py
│   ├── skeleton_io.py      # list[PoseResult] -> seq/mask, cắt cửa sổ (make_windows)
│   ├── features.py         # normalize_clip: chuẩn hóa clip — dùng CHUNG lúc train và lúc chạy
│   ├── graph.py  model_stgcn.py   # ST-GCN (đăng ký arch "stgcn")
│   ├── classifier.py       # FallClassifier + registry model + save_checkpoint (chung cho ST-GCN/LSTM)
│   ├── model_lstm.py       # LSTM (B1, arch "lstm")
│   ├── train_utils.py      # đọc dataset, tăng cường, chỉ số (numpy)
│   └── stgcn_inference.py  # tên cũ STGCNInference, giờ là lớp con của FallClassifier
├── web/                    # lớp giao diện, chỉ gọi lại src/
│   ├── engine.py           # tạo đối tượng từ cấu hình, chọn FileSource/StreamSource, đo latency, trích -> CSV
│   ├── server.py           # Flask: trang, MJPEG, API, tải về
│   ├── videos.py           # quét data/videos -> metadata.csv, kiểm tra nhãn/rò rỉ
│   ├── csv_tools.py        # đọc CSV, chuyển dạng dài, kiểm tra cho xử lý chuỗi
│   ├── dataset.py          # NPZ + metadata -> windows.npz
│   ├── dummy_backend.py    # backend pose giả (cài giao diện PoseBackend) để test
│   ├── config.py
│   └── templates/index.html
├── data/videos/{fall,adl}/ # video đầu vào
├── data/metadata.csv       # sinh bởi "Quét video"
├── output/<cấu hình>/      # ví dụ output/yolov8n-pose_10fps_filt/
│   ├── csv/<video_id>.csv
│   ├── npz/<video_id>.npz  # seq, mask, valid, t, frame_idx, fps, aspect (đầu vào dataset)
│   ├── dataset/windows.npz
│   └── _extract_summary.csv
├── models/yolov8n-pose.pt  models/best_stgcn.pt
├── data/dataset.npz        # dataset code_1 (train S2,S4 / val S1 / test S3)
├── tests/test_web.py  tests/test_model.py  tests/test_train.py
├── HUONG_DAN.md            # tài liệu gốc của newfilecode
└── requirements.txt
```

Tên thư mục kết quả gồm **model, FPS và có lọc hay không** (`..._filt` / `..._raw`), nên đổi cấu hình
không bao giờ ghi đè hay lẫn kết quả cũ.

## 5. Định dạng dữ liệu và kiểm tra CSV

CSV do `src/csv_export.py` sinh ra, **mỗi khung lấy mẫu một dòng** (kể cả khung không có người):

`video_id, frame_idx, t, valid, nose_x, nose_y, nose_conf, …, right_ankle_conf`

- `x, y` chuẩn hóa 0–1; khung `valid=0` và khớp thiếu: **ô để trống** (đọc vào là NaN), không phải 0.
- Khớp do bộ lọc "giữ tạm" có `x, y` nhưng `conf = 0`. Bước sau phải kiểm tra `conf`, không chỉ NaN.
- `t` theo lưới đều `frame_idx / target_fps` với video file; theo đồng hồ thật với webcam.

**Kiểm tra cho xử lý chuỗi** (link *kiểm tra* hoặc `run_web.py check-csv`), viết lại theo quy ước trên:

| Kiểm tra | Mức |
|---|---|
| Đủ cột, dựng được mảng `[T, 17, 3]` | Lỗi |
| `t` tăng dần, không trùng | Lỗi |
| Khung `valid=0` phải để trống toàn bộ khớp; không có khớp chỉ có x hoặc chỉ có y | Lỗi |
| `x, y` trong 0–1 | Lỗi |
| Video dài hơn cửa sổ, có ít nhất một cửa sổ đạt; mỗi file một `video_id` | Lỗi |
| Nhịp lấy mẫu đều, `frame_idx` liên tục | Cảnh báo |
| Tỉ lệ mất người, khoảng trống liên tiếp dài nhất, tỉ lệ khớp trống, confidence trung bình | Cảnh báo |

Tham số cửa sổ ở mục `check:` của `configs/web.yaml` (mặc định 2 s, bước 0,25 s, `max_missing` 0,4).

## 6. Cấu hình chính (`configs/web.yaml`)

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `source.target_fps` | 10 | FPS lấy mẫu (quy ước nhóm) |
| `pose.backend` | yolo | `yolo` (YoloBackend của `src/`) hoặc `dummy` |
| `pose.model_path` | models/yolov8n-pose.pt | Tên trần (như `yolo11n-pose.pt`) thì ultralytics tự tải |
| `pose.conf`, `pose.device` | 0.4, null | Ngưỡng phát hiện người; `cpu` / `cuda:0` |
| `pose.kp_conf`, `pose.min_valid_joints` | 0.35, 5 | Độ khắt khe của `valid` |
| `pose.use_tracker`, `pose.tracker` | true, bytetrack.yaml | `model.track()` giữ `track_id`; `false` = `predict()` |
| `pose.iou`, `pose.imgsz` | 0.5, null | NMS IoU; kích thước ảnh vào YOLO (320 để gần MaixCAM) |
| `filter.enabled` + các tham số | true | Bật SkeletonFilter khi trích. Làm mượt bằng One Euro: `min_cutoff` 1.5, `beta` 4.0 |
| `draw.min_conf`, `draw.show_bbox` | 0.35, true | Ngưỡng vẽ khớp; hộp bao + score trên ảnh MJPEG |
| `classifier.checkpoint` | auto | `auto` = models/best_lstm.pt, không có thì best_stgcn.pt; hoặc đường dẫn; null = tắt |
| `classifier.threshold`, `stride_frames`, `window_size` | 0.5, 5, null | P(ngã) ≥ ngưỡng → NGÃ; dự đoán mỗi 5 khung; cửa sổ lấy từ checkpoint |
| `dataset.win_sec`, `stride_sec`, `min_overlap` | 2.0, 0.5, 0.3 | Cắt cửa sổ khi tạo dataset (2 s @ 10 fps = 20 khung) |

## 7. API (cho ai muốn gọi từ script)

| Phương thức | Đường dẫn | Việc |
|---|---|---|
| GET | `/stream?source=…&skeleton=1&filter=1&realtime=1` | Luồng MJPEG |
| POST | `/api/stop` | Dừng luồng đang xem |
| GET | `/api/live`, `/api/status` | Số liệu luồng, trạng thái hệ thống |
| GET | `/api/videos` | Danh sách video trong metadata |
| POST | `/api/scan` | Quét video |
| POST | `/api/extract` body `{"source": "all"\|video_id\|đường dẫn, "overwrite": false}` | Tạo job trích |
| GET / POST | `/api/jobs`, `/api/jobs/<id>/cancel` | Tiến độ, hủy |
| GET | `/api/skeleton/<video_id>` | Dữ liệu CSV dạng JSON (NaN → null) |
| GET | `/api/check/<video_id>` | Báo cáo kiểm tra CSV |
| GET | `/video/<video_id>` | File video gốc (hỗ trợ Range để tua) |
| GET | `/download/<video_id>.csv[?format=long]`, `/download/_extract_summary.csv` | Tải về |
| GET | `/stream?...&classify=1` | Luồng MJPEG kèm phân loại ngã |
| GET | `/api/predict/<video_id>` | P(ngã) theo thời gian của video đã trích (`t`, `p`, `threshold`) |
| POST | `/api/build-dataset` | Tạo `windows.npz` |
| GET | `/compare` | Trang so sánh model (ST-GCN vs LSTM…) |
| POST | `/api/compare` body `{"checkpoints": [...], "data": "data/dataset.npz", "split": "test"}` | Chạy so sánh, trả JSON |
| GET | `/compare/export.csv` / `.json` / `.html` | Tải kết quả so sánh gần nhất (HTML = báo cáo một file) |
| GET / POST | `/api/classifier` body `{"checkpoint": "models/best_lstm.pt" \| "auto" \| null}` | Xem / đổi model phân loại |

## 8. Khác biệt so với fall_app

- Mỗi video trích ra cả CSV (xem/kiểm tra) và NPZ (giữ `aspect`, dùng để tạo dataset và dự đoán offline).
  CSV trích bằng bản cũ (chưa có NPZ) sẽ được trích lại khi bấm **Trích tất cả**.
- Quy ước "không có người" là **ô trống/NaN** (newfilecode), không phải 0 như fall_app.
- Chỉ có backend YOLO (và backend giả); chưa có ONNX/MoveNet như fall_app.
- Chỉ đọc file video, không đọc thư mục ảnh.
- Thêm so với fall_app: nút Quét video, bật/tắt SkeletonFilter khi xem trực tiếp, hủy job, tiến độ theo khung, tải CSV dạng dài và bảng tóm tắt.

## 9. Hạn chế đã biết

- Chưa chạy thử với YOLO thật, video thật có người, hay webcam thật. Đã kiểm thử đầu-cuối bằng backend giả,
  gồm cả `StreamSource` đọc một luồng MJPEG qua HTTP.
- Phần PyTorch (ST-GCN, `FallClassifier`, phát hiện ngã trên web) chưa được chạy trong môi trường bàn giao vì không cài được
  torch; test `test_classifier_end_to_end` tự bỏ qua khi thiếu torch. Chạy `python -m unittest discover tests` trên máy có torch để xác nhận.
- `pose.use_tracker: true` cần gói `lap` (ByteTrack); ultralytics thường tự cài lần đầu, nếu lỗi thì `pip install lap`.
- Trình duyệt không phát được video mã hóa `mp4v` / AVI-XVID (ô ghi chú dưới trình phát sẽ báo). Chuyển sang MP4 H.264
  (`ffmpeg -i in.avi -c:v libx264 out.mp4`) hoặc xem bằng ô nguồn + **Xem** (luồng MJPEG không phụ thuộc codec trình duyệt).
- Web chỉ theo dõi **một người** (giới hạn của `PoseEstimate` / `SkeletonFilter`).

## 10. B1: LSTM, huấn luyện và so sánh với ST-GCN

Đã có đủ, hướng dẫn từng bước (VSCode, cài đặt, train, đánh giá, chạy thật, lỗi thường gặp) ở
[`HUONG_DAN_B1.md`](HUONG_DAN_B1.md). Tóm tắt:

```bash
pip install torch && pip install -r requirements.txt
python -m unittest discover tests                                    # không còn test nào bị bỏ qua
python train.py --arch lstm                                          # data/dataset.npz -> models/best_lstm.pt
python evaluate.py models/best_stgcn.pt models/best_lstm.pt --device cpu
python main.py --source video.mp4 --checkpoint models/best_lstm.pt  # hoặc web: --config configs/code1.yaml
```

| Tệp | Vai trò |
|---|---|
| `src/model_lstm.py` | LSTM (`arch="lstm"`), cùng đầu vào `[N,3,20,17]` / đầu ra `[N,2]` với ST-GCN |
| `src/train_utils.py` | Đọc dataset (code_1 hoặc web), tăng cường, chỉ số, chọn ngưỡng (chỉ numpy) |
| `train.py` | Huấn luyện `--arch lstm` hoặc `--arch stgcn` (cách huấn luyện giữ như train.py của code_1); `--merge-val` gộp val vào train |
| `evaluate.py` | So sánh checkpoint trên cùng tập test: F1, recall, AUC, theo video, tham số, ms/cửa sổ; `--html` xuất báo cáo một file |
| `web/compare.py`, `web/templates/compare.html` | Trang **So sánh model** (`/compare`): biểu đồ, ROC, ma trận nhầm lẫn, khoảng tin cậy, từng video |
| `check_labels.py` | Tìm cửa sổ "nằm yên" bị gán nhãn ngã (nguyên nhân nằm bình thường bị báo ngã), `--fix` ghi dataset sạch |
| `data/dataset_clean.npz` | `dataset.npz` đã sửa 75 nhãn nằm yên → 0 ở train/val (test giữ nguyên) |
| `main.py` | Chạy trên video/webcam bằng đúng pipeline đã tạo dữ liệu (thay main.py của code_1) |
| `extract_skeleton.py`, `make_intervals.py`, `build_dataset.py` | Script tạo dataset của code_1 (giữ nguyên, chỉ đổi model mặc định) |
| `configs/code1.yaml` | Tham số pose giống lúc tạo `data/dataset.npz` |
| `data/dataset.npz`, `models/best_stgcn.pt` | Dataset và ST-GCN của code_1 |
| `.vscode/` | Interpreter, test, cấu hình F5 (train, so sánh, chạy video, web) |

Thêm kiến trúc khác sau này: một tệp `src/model_<tên>.py` có `@register_model("<tên>")`, thêm tên module vào
`_BUILTIN_MODULES` trong `src/classifier.py`, rồi `python train.py --arch <tên>`.
