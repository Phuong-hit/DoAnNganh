# Fall Web — phát hiện ngã từ skeleton

Phục vụ đồ án *Hệ thống phát hiện ngã dựa trên skeleton trên thiết bị biên*.

Video / webcam → YOLO-pose → lọc skeleton → cửa sổ 17 khung → một trong ba model: **B0** (luật),
**B1** (LSTM), **B2** (Shift-GCN) → NGÃ / KHÔNG NGÃ, chạy và xem trên giao diện web. Kết quả so sánh ở mục 10.

```
 Nguồn                       src/                                     Web
┌──────────────┐ khung ┌──────────────┐ PoseResult ┌────────────────┐ ┌──────────────────────────┐
│ FileSource   │──────►│ PoseEstimate │───────────►│ SkeletonFilter │►│ CSVExporter → CSV        │
│ StreamSource │       │ (YOLO-pose)  │            │ (tùy chọn)     │ │ + NPZ (skeleton_io)      │
└──────────────┘       └──────────────┘            └────────────────┘ │ SkeletonDraw → MJPEG     │
                                                           │          └──────────────────────────┘
                                                           ▼
                                      FallClassifier (cửa sổ 17 khung, B0 / B1 / B2) → NGÃ / không ngã
```

## 1. Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # có ultralytics (kéo theo PyTorch)
python -m unittest discover tests    # kiểm thử bằng pose giả, không cần model
```

Trọng số YOLO-pose không nằm trong git. Tải về `models/` trước khi chạy (`configs/v1.yaml` cần `yolo26m-pose.pt`,
`configs/web.yaml` cần `yolov8n-pose.pt`):

```bash
cd models
python -c "from ultralytics.utils.downloads import attempt_download_asset as a; a('yolo26m-pose.pt')"
cd ..
```

Model phân loại ngã đã có sẵn trong `models/`: `b0_rule_v1.json`, `b1_lstm.pt`, `shiftgcn_best.pt`.

## 2. Chạy

Luôn chạy từ thư mục gốc project (cùng cấp với `src/`).

```bash
python run_web.py --config configs/v1.yaml         # nên dùng: pose giống lúc tạo dataset, mở http://127.0.0.1:5000
python run_web.py                                  # cấu hình mặc định configs/web.yaml
python run_web.py web --host 0.0.0.0 --port 5000   # cho máy khác trong mạng LAN mở
python run_web.py --set pose.backend=dummy         # chạy thử giao diện khi chưa cài ultralytics
python run_web.py scan                             # quét video -> data/metadata.csv (giống nút "Quét video")
python run_web.py check-csv                        # kiểm tra mọi CSV của cấu hình hiện tại
python run_web.py build-dataset                    # NPZ + nhãn -> output/<cấu hình>/dataset/windows.npz
python run_web.py --set classifier.checkpoint=models/shiftgcn_best.pt   # đổi model phân loại (cũng đổi được trên web)
```

Quy trình thường dùng:

1. Chép video vào `data/videos/fall/` (video ngã) và `data/videos/adl/` (sinh hoạt thường).
2. Mở web, bấm **Quét video** → sinh `data/metadata.csv`. Mở file này điền tay `fall_start_s`, `fall_end_s`,
   `subject_id`, `split`… (các cột này được giữ nguyên khi quét lại).
3. Bấm **Trích tất cả** (hoặc *trích* ở từng dòng) → CSV ở `output/<cấu hình>/csv/<video_id>.csv`.
4. Bấm tên video để phát lại kèm khung xương; bấm *kiểm tra* để xem CSV có dùng được cho xử lý chuỗi không.
5. Bấm **Tạo dataset** (hoặc `run_web.py build-dataset`) → `windows.npz` để huấn luyện.
6. Chọn model ở ô **Model** trên web (`b0_rule_v1.json`, `b1_lstm.pt`, `shiftgcn_best.pt` hoặc `tắt`) → luồng trực tiếp có nhãn **FALL / no fall**, trình phát có dải P(ngã).

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
| Tạo dataset | NPZ đã trích + `fall_start_s/fall_end_s/split/subject_id` trong metadata → `windows.npz` (mục 9) |
| Tải về | CSV gốc (dạng rộng, đúng định dạng `CSVExporter`), CSV dạng dài (1 dòng/khớp), bảng `_extract_summary.csv` (số khung, tỉ lệ có người, `missing_ratio`, latency, FPS từng video) |

## 4. Cấu trúc thư mục

```
fall_web/
├── run_web.py              # điểm chạy web (web / scan / check-csv / build-dataset)
├── main.py                 # chạy trên video / webcam bằng cửa sổ OpenCV
├── test_tool.py            # Test Tool B0/B1/B2 trên bộ test chung
├── app_check.py            # chạy video thật qua pipeline App, so với bộ test
├── train.py  evaluate.py  tune_b0.py  tune_b1.py        # huấn luyện, so sánh, dò tham số B0, tinh chỉnh B1
├── extract_skeleton.py  build_dataset_v1.py             # video -> skeleton NPZ -> datasetv1
├── build_lying_negatives.py                             # thêm mẫu "nằm mà không ngã" -> datasetv1_lying
├── build_dataset.py  make_intervals.py  check_labels.py  calibrate_gate.py   # công cụ dataset cũ, cổng tốc độ
├── configs/v1.yaml         # nên dùng: pose giống lúc tạo datasetv1, cửa sổ 17 khung
├── configs/web.yaml  code1.yaml  v6.yaml                # các cấu hình pose khác
├── src/
│   ├── constants.py  pose_estimate.py  skeleton_filter.py  skeleton_draw.py
│   ├── video_process.py  webcam.py  csv_export.py
│   ├── skeleton_io.py      # list[PoseResult] -> seq/mask, cắt cửa sổ (make_windows)
│   ├── features.py         # normalize_clip: chuẩn hóa clip, dùng CHUNG lúc train và lúc chạy
│   ├── motion.py           # đặc trưng tốc độ + cổng tốc độ
│   ├── classifier.py       # FallClassifier + registry model (dùng chung cho mọi model)
│   ├── b0_rule.py          # B0: luật
│   ├── model_lstm.py       # B1: LSTM
│   ├── model_shiftgcn.py  shiftgcn_inference.py         # B2: Shift-GCN (giữ nguyên bản của nhóm)
│   ├── model_adapters.py   # nối Shift-GCN và B0 vào FallClassifier
│   ├── graph.py  model_stgcn.py  stgcn_inference.py     # ST-GCN (vẫn train được, hiện không có checkpoint)
│   ├── testset.py          # tạo / kiểm tra bộ test chung
│   └── train_utils.py  data_utils.py                    # đọc dataset, tăng cường, chỉ số
├── web/                    # giao diện, chỉ gọi lại src/
│   ├── engine.py           # tạo đối tượng từ cấu hình, chọn nguồn video, đo latency, trích -> CSV
│   ├── server.py           # Flask: trang, MJPEG, API, tải về
│   ├── compare.py          # trang so sánh model
│   ├── videos.py  csv_tools.py  dataset.py  frames.py  dummy_backend.py  config.py
│   └── templates/
├── models/                 # b0_rule_v1.json, b1_lstm.pt, shiftgcn_best.pt + trọng số YOLO-pose
├── data/
│   ├── testsets/v1_test/  v1_lying/                     # bộ test chung (không ghi đè)
│   ├── datasetv1.npz  datasetv1_lying.npz               # dataset 17 khung
│   ├── dataset.npz  dataset_clean.npz                   # dataset 20 khung cũ
│   └── videos/{fall,adl}/  metadata.csv                 # video của nhóm
├── output/<cấu hình>/      # csv/, npz/, dataset/windows.npz, _extract_summary.csv
├── output/test_runs/  output/app_check/                 # kết quả Test Tool và app_check
├── tests/
├── HUONG_DAN.md  HUONG_DAN_B1.md
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
| `classifier.checkpoint` | models/b1_lstm.pt | Đường dẫn model (`.pt` hoặc `models/b0_*.json`); null = tắt |
| `classifier.threshold`, `stride_frames`, `window_size` | null, 5, null | P(ngã) ≥ ngưỡng → NGÃ (null = ngưỡng lưu trong checkpoint); dự đoán mỗi 5 khung (`v1.yaml`: mỗi khung); cửa sổ lấy từ checkpoint |
| `classifier.speed_gate` | null | null = cổng tốc độ lưu trong checkpoint (chỉ B1 có), không có thì tắt |
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
| GET | `/compare` | Trang so sánh model |
| POST | `/api/compare` body `{"checkpoints": [...], "data": "data/dataset.npz", "split": "test"}` | Chạy so sánh, trả JSON |
| GET | `/compare/export.csv` / `.json` / `.html` | Tải kết quả so sánh gần nhất (HTML = báo cáo một file) |
| GET / POST | `/api/classifier` body `{"checkpoint": "models/b1_lstm.pt" \| null}` | Xem / đổi model phân loại |

## 8. Hạn chế đã biết

- **Nằm vẫn có thể bị báo ngã.** Dataset gốc gần như không có mẫu nằm mà không ngã. B1 đã được bổ sung bằng mẫu tạo thêm
  và cổng tốc độ nhưng còn báo nhầm 1/5 video ADL của nhóm (`sub3_08`). B2 (Shift-GCN) giữ nguyên bản của nhóm nên báo nhầm 4/5.
- **Skeleton trên App lệch dataset** khoảng 0,3 độ dài xương dù đã chỉnh pose giống `extract_skeleton.py`, có thể do bước lọc
  tạo `output/npz_filtered` chưa tái hiện được.
- Mẫu "nằm mà không ngã" là mẫu tạo thêm, video thật mới có 5 ADL của nhóm. Cần quay thêm để đánh giá chắc hơn.
- `test_compare_routes` trong `tests/test_train.py` thất bại nếu không có `models/best_stgcn.pt`.
- `pose.use_tracker: true` cần gói `lap` (ByteTrack); ultralytics thường tự cài lần đầu, nếu lỗi thì `pip install lap`.
- Trình duyệt không phát được video mã hóa `mp4v` / AVI-XVID (ô ghi chú dưới trình phát sẽ báo). Chuyển sang MP4 H.264
  (`ffmpeg -i in.avi -c:v libx264 out.mp4`) hoặc xem bằng ô nguồn + **Xem** (luồng MJPEG không phụ thuộc codec trình duyệt).
- Web chỉ theo dõi **một người** (giới hạn của `PoseEstimate` / `SkeletonFilter`).

## 9. Huấn luyện và công cụ

Hướng dẫn chi tiết ở [`HUONG_DAN_B1.md`](HUONG_DAN_B1.md) (viết cho dataset 20 khung cũ; lệnh train vẫn dùng được).

```bash
python build_lying_negatives.py                      # tạo data/datasetv1_lying.npz
python tune_b1.py --data data/datasetv1_lying.npz    # thử nhiều cấu hình LSTM, chọn theo val
python train.py --arch lstm --data data/datasetv1_lying.npz --out models/x.pt   # train một cấu hình
python tune_b0.py --data data/datasetv1.npz --name b0_rule_v1                   # dò tham số B0 trên train
```

| Tệp | Vai trò |
|---|---|
| `train.py` | Huấn luyện `--arch lstm` hoặc `--arch stgcn`; ngưỡng chọn trên val, lưu trong checkpoint |
| `tune_b1.py` | Nhiều cấu hình × nhiều seed, chọn theo val; `--reselect` chọn lại không train |
| `evaluate.py`, trang `/compare` | So sánh checkpoint trên một dataset: F1, AUC, ROC, ms/cửa sổ, báo cáo HTML |
| `src/motion.py`, `calibrate_gate.py` | Cổng tốc độ: chỉ xác nhận ngã khi vai hạ / thân xoay đủ nhanh |
| `check_labels.py` | Tìm cửa sổ "nằm yên" bị gán nhãn ngã trong dataset cũ, `--fix` ghi dataset sạch |
| `extract_skeleton.py`, `build_dataset_v1.py` | Video → skeleton NPZ → dataset 17 khung |

Thêm kiến trúc mới: một tệp `src/model_<tên>.py` có `@register_model("<tên>")`, thêm tên module vào
`_BUILTIN_MODULES` trong `src/classifier.py`, rồi `python train.py --arch <tên>`.

## 10. Bộ test chung, Test Tool B0/B1/B2, kết nối App

Cả nhóm đánh giá trên **cùng một bộ test** `data/testsets/v1_test/` (split test của `data/datasetv1.npz`:
subject SBJ13, SBJ17, SBJ28; 441 cửa sổ 17 khung @ 10 fps; 147 video, 82 ngã). `manifest.json` ghi nguồn,
quy ước nhãn (0 = ADL, 1 = Fall), định dạng và mã băm nội dung; `load_testset()` báo lỗi nếu dữ liệu bị sửa.
`data/testsets/v1_lying/` là bộ trên cộng 78 cửa sổ tạo thêm (nằm yên, nằm xuống chậm, nhãn 0), vì `v1_test` không có ADL nằm.

| Vai trò | Model | File |
|---|---|---|
| B0 | Luật (Phúc), tham số dò trên train của datasetv1 | `models/b0_rule_v1.json` (`tune_b0.py --data data/datasetv1.npz --name b0_rule_v1`) |
| B1 | LSTM (dropout 0.5) train trên `data/datasetv1_lying.npz` (`build_lying_negatives.py`: thêm mẫu nằm yên / nằm xuống chậm), cổng tốc độ vai 1.6 thân/s, xoay 70 độ/s lưu trong checkpoint | `models/b1_lstm.pt` |
| B2 | Shift-GCN, giữ nguyên bản của nhóm (`src/model_shiftgcn.py`, arch `shiftgcn`), không có cổng tốc độ | `models/shiftgcn_best.pt` |

```bash
python test_tool.py                                   # B0/B1/B2 trên v1_test -> output/test_runs/<thời gian>/
python test_tool.py --testset data/testsets/v1_lying  # bộ test có thêm tình huống nằm mà không ngã
python test_tool.py --extra tên=models/x.pt           # thêm model để so sánh
python test_tool.py --make-testset v2_test --data data/x.npz --split test   # tạo bộ test mới (không ghi đè bộ cũ)
python app_check.py                                   # chạy video thật qua pipeline App, so với bộ test
python run_web.py --config configs/v1.yaml            # web với pose giống lúc trích datasetv1 + B1
```

Test Tool ghi `report.md` (bảng P/R/F1 theo cửa sổ và theo video, thời gian xử lý, phân tích báo nhầm / bỏ sót
theo hành động và hướng camera), `predictions.csv` (nhãn thật + dự đoán từng cửa sổ), `video_predictions.csv`,
`errors.csv`, `by_activity.csv`, `comparison.csv`, `summary.json`. `FallClassifier` đọc được checkpoint Shift-GCN của nhóm
(`model_state_dict`, `num_frames`, ...) và file JSON của B0, nên web và `main.py` dùng chung một đường.
`build_dataset_v1.py` là script tạo `datasetv1.npz` (cửa sổ 17 khung, stride 1, nhãn theo cả video).

**Kết quả hiện tại:**

| Model | F1 `v1_test` | F1 `v1_lying` | ms/cửa sổ (CPU) | Video nhóm: ADL báo nhầm | Video nhóm: ngã bắt được |
|---|---|---|---|---|---|
| B0 | 0.923 | 0.906 | 0.2 | 3/5 | 3/4 |
| B1 | 0.979 | 0.979 | 1.6 | 1/5 | 4/4 |
| B2 | 0.984 | 0.907 | 3.2 | 4/5 | 4/4 |

`v1_lying` dùng cùng cách tạo mẫu với lúc train B1 nên có lợi cho B1. Video nhóm chỉ có 9 video (5 ADL, 4 ngã).
