# Fall Web — giao diện web cho pipeline pose (newfilecode)

Phục vụ đồ án *Hệ thống phát hiện ngã dựa trên skeleton trên thiết bị biên*.

- **Xử lý** (nguồn video → ước lượng tư thế → lọc → ghi CSV / vẽ) dùng **nguyên vẹn** các tệp trong `src/`
  của newfilecode, không sửa dòng nào. Logic từng tệp xem ở [`HUONG_DAN.md`](HUONG_DAN.md).
- **Giao diện web** (`web/`) có các chức năng như giao diện web của fall_app, viết lại để gọi các lớp
  `FileSource`, `StreamSource`, `PoseEstimate`, `SkeletonFilter`, `SkeletonDraw`, `CSVExporter` của `src/`.

```
 Nguồn                       src/ (giữ nguyên)                                   Web
┌──────────────┐ khung ┌──────────────┐ PoseResult ┌────────────────┐ ┌──────────────────────────┐
│ FileSource   │──────►│ PoseEstimate │───────────►│ SkeletonFilter │►│ CSVExporter → CSV        │
│ StreamSource │       │ (YOLO-pose)  │            │ (tùy chọn)     │ │ SkeletonDraw → MJPEG     │
└──────────────┘       └──────────────┘            └────────────────┘ └──────────────────────────┘
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
```

Quy trình thường dùng:

1. Chép video vào `data/videos/fall/` (video ngã) và `data/videos/adl/` (sinh hoạt thường).
2. Mở web, bấm **Quét video** → sinh `data/metadata.csv`. Mở file này điền tay `fall_start_s`, `fall_end_s`,
   `subject_id`, `split`… (các cột này được giữ nguyên khi quét lại).
3. Bấm **Trích tất cả** (hoặc *trích* ở từng dòng) → CSV ở `output/<cấu hình>/csv/<video_id>.csv`.
4. Bấm tên video để phát lại kèm khung xương; bấm *kiểm tra* để xem CSV có dùng được cho xử lý chuỗi không.

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
| Tải về | CSV gốc (dạng rộng, đúng định dạng `CSVExporter`), CSV dạng dài (1 dòng/khớp), bảng `_extract_summary.csv` (số khung, tỉ lệ có người, `missing_ratio`, latency, FPS từng video) |
| Tải toàn bộ khung | Nút **Tải toàn bộ khung (ZIP)** dưới trình phát (bật khi video đã có CSV). Tạo một file ZIP chứa mọi khung của video dạng JPEG, mỗi khung đã vẽ skeleton lấy từ CSV và ghi chú `khung` + `t` ở góc dưới. Tên file: `frame_<chỉ số>_t<giây>.jpg` |

## 4. Cấu trúc thư mục

```
fall_web/
├── run_web.py              # điểm chạy (web / scan / check-csv)
├── configs/web.yaml        # mọi thiết lập: FPS, model, ngưỡng, bộ lọc, tham số kiểm tra
├── src/                    # newfilecode — GIỮ NGUYÊN, xem HUONG_DAN.md
│   ├── constants.py  pose_estimate.py  skeleton_filter.py  skeleton_draw.py
│   └── video_process.py  webcam.py  csv_export.py
├── web/                    # lớp giao diện, chỉ gọi lại src/
│   ├── engine.py           # tạo đối tượng từ cấu hình, chọn FileSource/StreamSource, đo latency, trích -> CSV
│   ├── server.py           # Flask: trang, MJPEG, API, tải về
│   ├── videos.py           # quét data/videos -> metadata.csv, kiểm tra nhãn/rò rỉ
│   ├── csv_tools.py        # đọc CSV, chuyển dạng dài, kiểm tra cho xử lý chuỗi
│   ├── dummy_backend.py    # backend pose giả (cài giao diện PoseBackend) để test
│   ├── config.py
│   └── templates/index.html
├── data/videos/{fall,adl}/ # video đầu vào
├── data/metadata.csv       # sinh bởi "Quét video"
├── output/<cấu hình>/      # ví dụ output/yolov8n-pose_10fps_filt/
│   ├── csv/<video_id>.csv
│   └── _extract_summary.csv
├── models/yolov8n-pose.pt
├── tests/test_web.py
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
| `pose.kp_conf`, `pose.min_valid_joints` | 0.3, 5 | Độ khắt khe của `valid` |
| `filter.enabled` + các tham số | true | Bật SkeletonFilter khi trích; các tham số giống bảng mục 4 của HUONG_DAN |
| `draw.min_conf` | 0.3 | Ngưỡng vẽ khớp (cả ảnh MJPEG và lớp vẽ trên trình phát) |

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
| GET | `/frames/<video_id>.zip` | ZIP mọi khung đã vẽ skeleton |

## 8. Khác biệt so với fall_app

- Không có file `.npz`: newfilecode chỉ xuất CSV, nên CSV là định dạng lưu chính; trình phát và kiểm tra đều đọc CSV.
- Quy ước "không có người" là **ô trống/NaN** (newfilecode), không phải 0 như fall_app.
- Chỉ có backend YOLO (và backend giả); chưa có ONNX/MoveNet như fall_app.
- Chỉ đọc file video, không đọc thư mục ảnh.
- Thêm so với fall_app: nút Quét video, bật/tắt SkeletonFilter khi xem trực tiếp, hủy job, tiến độ theo khung, tải CSV dạng dài và bảng tóm tắt.

## 9. Hạn chế đã biết

- Chưa chạy thử với YOLO thật, video thật có người, hay webcam thật. Đã kiểm thử đầu-cuối bằng backend giả,
  gồm cả `StreamSource` đọc một luồng MJPEG qua HTTP.
- `YoloBackend` không nhận `imgsz`, nên ultralytics dùng mặc định 640; độ trễ đo được không đại diện cho cấu hình 320 dự kiến trên MaixCAM.
- Trình duyệt không phát được video mã hóa `mp4v` / AVI-XVID (ô ghi chú dưới trình phát sẽ báo). Chuyển sang MP4 H.264
  (`ffmpeg -i in.avi -c:v libx264 out.mp4`) hoặc xem bằng ô nguồn + **Xem** (luồng MJPEG không phụ thuộc codec trình duyệt).
- Web chỉ theo dõi **một người** (giới hạn của `PoseEstimate` / `SkeletonFilter`).
