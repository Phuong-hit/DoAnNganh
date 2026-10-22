# Hướng dẫn bàn giao: pipeline pose cho phát hiện ngã

Tài liệu này giải thích **logic của từng tệp và lý do thiết kế**, để người nhận có thể hiểu và làm tiếp mà không phải đoán. Cách cài và chạy nhanh nằm ở `README.md`.

> **Tình trạng:** toàn bộ mã đã được chạy thử đầu-cuối bằng backend giả và dữ liệu tổng hợp. **Chưa** kiểm thử với model YOLO thật, video thật hay camera thật. Các ngưỡng của bộ lọc là giá trị khởi đầu, chưa hiệu chỉnh trên video có cú ngã thật. Xem mục 7 để biết việc nên làm đầu tiên.

---

## 1. Bức tranh tổng thể

```
 Nguồn ảnh                 Ước lượng            Lọc theo thời gian        Đầu ra
┌────────────┐  khung   ┌──────────────┐ Pose  ┌─────────────────┐ Pose  ┌─────────────┐
│ FileSource │ ───────► │ PoseEstimate │ ────► │ SkeletonFilter  │ ────► │ CSVExporter │  → output/csv
│ StreamSrc  │ (t, idx) │ (YOLO-pose)  │Result │ (làm mượt)      │Result │ SkeletonDraw│  → video / ảnh
└────────────┘          └──────────────┘       └─────────────────┘       └─────────────┘
```

Mỗi khung đi qua đúng một chuỗi: **lấy mẫu → ước lượng → lọc → ghi/vẽ**. `main.py` chỉ nối các khối này lại và đọc cấu hình; mọi logic nằm trong `src/`.

Nguyên tắc xuyên suốt: **mọi khung được lấy mẫu đều có đúng một kết quả** (`PoseResult`), kể cả khi không có người. Không bao giờ "bỏ khung" im lặng, vì làm vậy sẽ mất thông tin thời gian và làm `missing_ratio` sai.

---

## 2. Quy ước dữ liệu (đọc kỹ phần này trước)

Đây là hợp đồng giữa các tệp. Sửa quy ước nào thì phải sửa ở mọi nơi dùng nó.

**`PoseResult`** (định nghĩa trong `src/pose_estimate.py`):

| Trường | Ý nghĩa |
|---|---|
| `frame_idx` | Chỉ số khung **sau lấy mẫu** (0, 1, 2…), không phải chỉ số khung gốc của video |
| `t` | Thời gian tính bằng giây |
| `valid` | `True` nếu tìm thấy người đủ tin cậy; `False` nếu không |
| `joints` | Mảng numpy `(17, 3)`, mỗi dòng là `[x, y, conf]` |
| `width`, `height` | Kích thước khung gốc (pixel), dùng để sửa tỉ lệ khung hình |
| `bbox`, `track_id` | Hộp bao (chuẩn hóa 0–1) và id theo dõi nếu có |

**Quy tắc của `joints`:**
- `x`, `y` **chuẩn hóa 0–1** theo (chiều rộng, chiều cao) khung gốc. Lý do: kết quả không phụ thuộc độ phân giải video, nên ngưỡng và đặc trưng dùng lại được giữa các video khác nhau.
- Khớp không quan sát được: `x = y = NaN`, `conf = 0`. Khung `valid=False`: **cả 17 khớp** là NaN. Khi tính toán phải dùng `np.isfinite` (hoặc `np.nan*` như `np.nanmedian`), đừng so sánh thẳng với NaN.
- Vì x và y được chuẩn hóa độc lập, **khoảng cách trực tiếp trong không gian 0–1 bị méo** nếu khung không vuông. Muốn tính khoảng cách đúng, nhân `dx` với `aspect = width/height` (thuộc tính `PoseResult.aspect`). Bộ lọc đã làm việc này.

**Thời gian `t`:**
- Video file: lưới đều `t = frame_idx / out_fps` (ví dụ 0.0, 0.1, 0.2… ở 10 fps).
- Webcam: đồng hồ thật tại lúc chụp, tính từ khung đầu tiên (nên có thể không đều nếu máy xử lý chậm).

**Thứ tự 17 keypoint** là chuẩn COCO, định nghĩa duy nhất ở `src/constants.py`. `joints[i]` ứng với `KEYPOINT_NAMES[i]`.

---

## 3. Giải thích từng tệp

### 3.1 `src/constants.py`: nguồn sự thật về keypoint và cách nối

Chỉ chứa hằng số, không có logic. Gồm: `KEYPOINT_NAMES` (17 tên), `KEYPOINT_INDEX` (tên → chỉ số), `SKELETON_EDGE_NAMES`/`SKELETON_EDGES` (các cạnh để **vẽ**), `BONE_NAMES`/`BONES` (các "xương" để **kiểm tra độ dài** trong bộ lọc), và các nhóm khớp (`HEAD`, `SHOULDERS`, `HIPS`, `KNEES`, `ANKLES`, `TORSO`) dành cho việc tính đặc trưng ngã sau này.

- Danh sách cạnh viết bằng **tên** cho dễ đọc, bản **chỉ số** được sinh tự động từ đó. Muốn đổi cách nối, chỉ sửa danh sách tên.
- `SKELETON_EDGES` (18 cạnh, gồm cả đầu và vai) và `BONES` (10 cạnh, chỉ tay-chân-thân) **khác nhau có chủ ý**: bộ lọc không kiểm tra độ dài ở đầu hay vai vì chúng thay đổi nhiều theo góc nhìn.
- Có `assert` ở cuối tệp: tên trùng hoặc sai số lượng sẽ báo lỗi ngay khi import.

### 3.2 `src/pose_estimate.py`: từ ảnh ra `PoseResult`

**Các lớp:**
- `PoseResult`: kiểu dữ liệu chuẩn (mục 2). Có `PoseResult.empty(...)` để tạo kết quả "không có người".
- `Detection`: một người do backend trả về, còn ở **đơn vị pixel** (`kpts_px`, `conf`, `bbox_px`).
- `PoseBackend` (giao diện) và `YoloBackend` (cài đặt dùng Ultralytics).
- `PoseEstimate`: lớp chính, biến danh sách `Detection` thành đúng một `PoseResult`.

**Luồng `process_frame(frame, frame_idx, t)`:**
1. Gọi `backend.detect(frame)` lấy danh sách người.
2. **Chọn một người** (`_select`): nếu đang bám một người ở khung trước, chọn người có **IoU hộp bao lớn nhất** với hộp trước đó, miễn là IoU ≥ `track_iou` (0.1). Nếu không có hoặc IoU quá thấp, chọn người có **hộp bao lớn nhất**. Lý do: bản cũ luôn lấy người đầu tiên nên có thể nhảy giữa các người khi có nhiều người trong khung.
3. **Kiểm tra chất lượng:** đếm số khớp có `conf ≥ kp_conf` (0.3). Dưới `min_valid_joints` (5) thì coi như không có người → trả `PoseResult.empty(...)`, `valid=False`.
4. **Chuẩn hóa:** chia x cho `width`, y cho `height`, cắt về [0, 1]. Khớp có `conf ≤ 0` hoặc đúng `(0, 0)` được coi là thiếu (YOLO trả `(0, 0)` cho khớp không có) → NaN.
5. Cập nhật hộp bao đang bám; nếu mất người thì tăng bộ đếm `_lost`, quá `lost_reset_frames` (30 khung) thì quên người cũ.

**Vì sao dùng `predict()` thay vì `track()` của Ultralytics:** `track(persist=True)` giữ trạng thái ngầm bên trong model, không reset được khi sang video mới. Ở đây việc bám người tự làm bằng IoU, nên `reset()` đơn giản và chắc chắn.

**Backend lazy:** `ultralytics` chỉ được import khi tạo `YoloBackend`. Muốn dùng ONNX/MoveNet/mô hình khác, viết một lớp có `detect(frame_bgr) -> list[Detection]` và `close()`, rồi truyền `PoseEstimate(backend=...)`. `model_path` là tên trần (như `yolo11n-pose.pt`) thì Ultralytics tự tải; có thư mục thì phải tồn tại.

**Gọi `reset()` mỗi khi sang video mới** để không bám nhầm người của video trước (`main.py` đã làm).

### 3.3 `src/skeleton_filter.py`: làm mượt và loại nhiễu theo thời gian

Đây là tệp phức tạp nhất. Nhận `PoseResult`, trả `PoseResult` **mới** (không sửa đầu vào). Mục tiêu: giảm rung và loại các khớp nhảy bất thường, **nhưng không phá những chuyển động thật như cú ngã**.

**Trạng thái giữ giữa các khung:** `prev_xy` (vị trí đã lọc gần nhất của từng khớp), `hist` (lịch sử `history_size=3` điểm của từng khớp), `missing` (số khung liên tiếp khớp đó không hợp lệ), `_gap` (số khung `valid=False` liên tiếp).

**Các bước cho mỗi khung hợp lệ:**
1. **Khung `valid=False`:** trả nguyên, tăng `_gap`; quá `max_gap_frames` (15 khung, tức 1.5 giây ở 10 fps) thì `reset()` toàn bộ trạng thái.
2. **Mặt nạ `ok`:** khớp có tọa độ hữu hạn và `conf ≥ min_conf` (0.3).
3. **Loại xương dài bất thường** (`_drop_long_bones`): ước lượng "độ dài xương điển hình" bằng **trung vị** các xương trong `BONES` có cả hai đầu hợp lệ. Xương nào dài hơn `max_bone_ratio` (2.5) lần mức điển hình thì loại khớp có conf thấp hơn. Cố ý **không ép một tư thế đứng**: người nằm hay co người vẫn hợp lệ.
4. **Ngưỡng nhảy:** `jump_limit = max(min_jump, jump_ratio × độ dài xương điển hình)` = tối đa của 0.04 và 0.65 × xương. Đơn vị là chiều cao khung (0–1), khoảng cách đã sửa theo `aspect`.
5. **Chuyển động chung:** trung vị độ dịch của các khớp (cần ≥ 3 khớp) giữa hai khung, để cả người di chuyển không bị coi là từng khớp tự nhảy.
6. **Phát hiện "nhảy":** khớp có độ lệch còn lại (sau khi trừ chuyển động chung) lớn hơn `jump_limit` là *jumper*. **Chỉ kiểm tra khớp vừa được quan sát khung trước** (`missing == 0`).
7. **Quy tắc quan trọng, "sụp đổ" = chuyển động thật:** nếu ≥ `collapse_ratio` (50%) số khớp ứng viên cùng "nhảy" trong một khung, đó là cả cơ thể thay đổi nhanh (ngã, xoay người), nên **chấp nhận tất cả** và "snap" về quan sát mới. Chuyển động chung chỉ bù được phép **tịnh tiến**, không bù được **xoay**, nên cú ngã sẽ làm hàng loạt khớp lệch cùng lúc.
8. **Khớp tìm lại sau khi mất** (`missing > 0`): được chấp nhận và snap, xóa lịch sử, **không trộn** với vị trí cũ đã lỗi thời.
9. **Làm mượt:** trung vị của lịch sử 3 điểm, rồi EMA với `alpha` thích nghi: bắt đầu từ `alpha=0.5` và tăng dần về 1 khi chuyển động của khớp tiến gần `jump_limit` (chuyển động lớn thì gần như không làm mượt, tránh trễ).
10. **Khớp không hợp lệ:** giữ nguyên vị trí cũ với `conf = 0` tối đa `max_hold_frames` (3) khung (báo "không phải quan sát mới"), sau đó đặt NaN và xóa lịch sử.

**Ba lỗi của bản cũ đã được sửa ở đây:** (1) mất hàng loạt khớp sau cú ngã (bước 7); (2) kéo khung về vị trí cũ (bước 8–9); (3) phụ thuộc độ phân giải do tính theo pixel (đơn vị 0–1 và `aspect`).

**Lưu ý khi dùng kết quả:** khớp được "giữ" có tọa độ nhưng `conf = 0`. Phần phía sau (đặc trưng ngã) phải kiểm tra `conf` chứ không chỉ kiểm tra NaN. Bộ lọc hiện chỉ xử lý **một người**.

### 3.4 `src/video_process.py`: nguồn video file và lấy mẫu `target_fps`

Chứa `FramePacket` (gói `idx, t, frame, src_idx`; có `__iter__` nên có thể viết `for idx, t, frame in ...`) và `FileSource` (tên cũ `VideoProcess` vẫn dùng được).

**Logic lấy mẫu trong `get_frames()`:**
- `out_fps = target_fps` nếu `target_fps` nhỏ hơn fps gốc; ngược lại `out_fps = fps gốc` (**không tăng mẫu**, không nhân bản khung).
- Mỗi khung gốc có thời điểm `ts = src_idx / src_fps`. Khung được phát ra khi `ts + nửa chu kỳ gốc ≥ mốc lưới kế tiếp`, tức chọn **khung gốc gần nhất với từng mốc** `k / out_fps`.
- Khung thứ k có `idx = k` và `t = k / out_fps` (lưới đều). Thời điểm thật của khung gốc có thể lệch `t` tối đa nửa chu kỳ gốc (vài mili-giây).
- Ví dụ: video 30 fps dài 3 giây → 90 khung gốc → 30 khung ra, `t = 0.0, 0.1, …, 2.9`.

**Hạn chế:** `ts` tính bằng `chỉ số / fps`, đúng với video fps cố định. Video có fps thay đổi (VFR) sẽ có sai số.
Thư mục `output/frames/<tên>/` chỉ được tạo khi thật sự gọi `save_frame` (tránh thư mục rỗng). `max_frames` để chạy thử nhanh.

### 3.5 `src/webcam.py`: nguồn camera/stream thời gian thực (`StreamSource`)

Bài toán: camera cấp khung nhanh hơn khả năng xử lý, nếu cứ đọc tuần tự thì khung cũ dồn lại và hình hiển thị bị trễ.

**Cách làm:**
- Một **luồng đọc riêng** (`_reader`) đọc camera liên tục và chỉ giữ **khung mới nhất** (`_latest`), khung cũ bị ghi đè. Dùng `threading.Condition` để đánh thức bên tiêu thụ.
- `get_frames()` phát khung theo lịch `next_emit` với chu kỳ `1/target_fps`. Khung đến sớm hơn lịch bị bỏ. Sau mỗi lần phát, `next_emit = max(next_emit + chu kỳ, thời điểm khung vừa phát)`, nghĩa là **không dồn nợ**: nếu xử lý chậm, khung kế tiếp là khung mới nhất chứ không phải để bù.
- `t` = `time.monotonic()` lúc chụp trừ đi lúc chụp khung đầu. Dùng đồng hồ thật nên đúng khi camera không ổn định fps.

**Điều kiện dừng:** `stop()`, `max_frames`, `max_seconds`, hoặc đọc lỗi liên tiếp `read_fail_limit` (30) lần (camera bị rút/mất tín hiệu). `release()` dừng luồng và giải phóng camera. Tên cũ `Webcam` vẫn dùng được.

**Lưu ý:** nếu cấp một **file video** cho `StreamSource`, luồng đọc chạy nhanh hơn thời gian thật nên chỉ nhận được rất ít khung. Chế độ này dành cho camera/stream thật; video file nên dùng `FileSource`.

### 3.6 `src/csv_export.py`: ghi CSV

**Mỗi khung lấy mẫu đúng một dòng.** Cột: `video_id, frame_idx, t, valid`, rồi với mỗi khớp `<tên>_x, <tên>_y, <tên>_conf` (17 × 3 cột).
- Khung `valid=False` hoặc khớp NaN: ô **để trống** (không ghi 0, vì 0 là một tọa độ hợp lệ).
- Làm tròn: `t` 3 chữ số, x/y 5 chữ số, conf 4 chữ số. Mã hóa `utf-8-sig` để Excel mở đúng.
- **`missing_ratio`** = (số khung `valid=0` + số khung bị nhảy cóc) / tổng khung. "Nhảy cóc" là khoảng trống trong `frame_idx` mà không có dòng nào. Đây là chỗ bản cũ sai: nó bỏ khung không có người nên luôn báo 0.0.
- Dùng như context manager: `with CSVExporter(video_id, output_dir) as out: out.write(pose)`.

Schema này là **giả định** về `skeleton_io.py` của FallApp. Nếu FallApp dùng tên hoặc thứ tự cột khác, sửa `META_COLUMNS` và `_joint_columns()`.

### 3.7 `src/skeleton_draw.py`: vẽ skeleton

`SkeletonDraw.draw(frame, pose)` trả **bản sao** ảnh đã vẽ (không sửa ảnh gốc).
- Tọa độ 0–1 được **nhân với chiều rộng/cao của khung đang vẽ** (không dùng `pose.width/height`), nên vẽ đúng cả khi ảnh đã bị resize.
- Khớp chỉ được vẽ nếu tọa độ hữu hạn **và** `conf ≥ min_conf` (0.3). `conf` NaN được coi như 0 nên không còn lỗi so sánh với `None` của bản cũ.
- Nối các cạnh trong `SKELETON_EDGES` (xanh lá), khớp là chấm đỏ. Khung `valid=False` hiện chữ "NO PERSON" màu đỏ ở góc trái trên.

### 3.8 `main.py`: điểm chạy và cấu hình

Phần đầu tệp là khối **CẤU HÌNH**; sửa tại đó, không cần tham số dòng lệnh.

**Luồng:** `main()` tạo `SkeletonFilter`, `SkeletonDraw`, `PoseEstimate`, rồi gọi `run_video_mode` hoặc `run_webcam_mode` tùy `MODE`. Cả hai gọi chung `process_source(src, ...)`, và đây là nơi pipeline thật sự chạy: mỗi khung → `estimator.process_frame` → `flt.smooth` → `csv_out.write` → (vẽ một lần, dùng cho hiển thị, video overlay và ảnh lưu).

**Điểm cần nhớ:**
- `process_source` gọi `estimator.reset()` và `flt.reset()` ở đầu mỗi nguồn, để trạng thái không rò từ video này sang video khác.
- Ảnh đã vẽ (`vis`) chỉ được tạo khi có nơi dùng: hiển thị, `SAVE_VIDEO_*` hoặc `SAVE_FRAMES` kèm `FRAMES_WITH_POSE`.
- Tên tệp kết quả lấy từ **tên video** (`01.mp4` → `01.csv`), nên chạy hai video trùng tên ở hai thư mục khác nhau sẽ ghi đè nhau. Khi chạy nhiều subject, đổi `OUTPUT_DIR` mỗi lần hoặc đưa tên thư mục cha vào `video_id`.
- `VIDEO_PATH` nhận file hoặc thư mục, nhưng chỉ lấy video ngay trong thư mục, không đi vào thư mục con.

---

## 4. Bảng tham số quan trọng

| Tham số | Ở đâu | Mặc định | Khi nào đổi |
|---|---|---|---|
| `TARGET_FPS` | `main.py` | 10 | Chỉ đổi nếu cả nhóm đổi quy ước; ngưỡng/đặc trưng phụ thuộc vào đây |
| `CONF` | `main.py` | 0.4 | Hạ (0.25) nếu hay "NO PERSON"; nâng nếu có người ảo |
| `kp_conf`, `min_valid_joints` | `PoseEstimate` | 0.3 / 5 | Điều chỉnh độ khắt khe của `valid` |
| `min_conf`, `alpha`, `max_hold_frames` | `SkeletonFilter` | 0.3 / 0.5 / 3 | Tăng `alpha` nếu bị trễ; giảm nếu còn rung |
| `jump_ratio`, `min_jump` | `SkeletonFilter` | 0.65 / 0.04 | Nếu vẫn loại nhầm khớp khi chuyển động nhanh thì tăng |
| `max_bone_ratio` | `SkeletonFilter` | 2.5 | Tăng nếu tư thế nằm/co bị loại khớp nhầm |
| `collapse_ratio` | `SkeletonFilter` | 0.5 | Giảm nếu cú ngã vẫn làm rớt nhiều khớp |
| `min_conf` | `SkeletonDraw` | 0.3 | Chỉ ảnh hưởng hiển thị |

---

## 5. Cách mở rộng

- **Đổi cách nối khớp / thêm nhóm khớp:** sửa `src/constants.py`.
- **Dùng mô hình khác:** viết backend (mục 3.2).
- **Thêm bộ phát hiện ngã:** thêm một bước sau `flt.smooth(pose)` trong `process_source`, nhận `PoseResult` đã lọc. Gợi ý đặc trưng: góc thân (vai → hông), tỉ lệ chiều cao/rộng của hộp bao, vận tốc hạ thấp của hông, thời gian nằm yên. Dùng nhóm khớp trong `constants.py` và nhớ xử lý NaN, `conf = 0` và `aspect`.
- **Nhiều người:** hiện chỉ chọn và theo dõi một người. Muốn nhiều người cần `PoseResult` chứa danh sách và bộ lọc có trạng thái riêng cho từng `track_id`.
- **Hiển thị tiến độ:** `main.py` chưa in tiến độ khi chạy, video dài sẽ có vẻ "đứng yên". Có thể in mỗi N khung trong `process_source`.

---

## 6. Lỗi thường gặp

| Hiện tượng | Nguyên nhân / cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'pose_estimate'` | Tệp trong `src/` còn là bản import tuyệt đối. Phải là `from .pose_estimate import ...` |
| `No module named 'src'` | Chạy từ sai thư mục. Chạy `python main.py` ngay trong `App/`; tạo `src/__init__.py` nếu cần |
| `cannot import name 'KEYPOINT_NAMES' from 'src.constants'` | `constants.py` là tệp cũ/tự tạo, chưa ghi đè bằng bản đúng |
| `FileNotFoundError` khi nạp model | `MODEL_PATH` có thư mục nhưng không tồn tại (đường dẫn tương đối tính từ nơi chạy) |
| Thư mục `frames` rỗng | `SAVE_FRAMES` đang `False`, hoặc `video_process.py`/`webcam.py` còn bản cũ |
| Ảnh không có skeleton | `FRAMES_WITH_POSE = False`, hoặc khung đó `valid=False` ("NO PERSON") |
| "Lỗi camera: Không mở được nguồn stream" | Camera đang bị ứng dụng khác dùng, hoặc sai `WEBCAM_SOURCE` |
| Ra ít khung hơn dự kiến ở webcam | Máy không kịp `TARGET_FPS`; bỏ hiển thị, dùng GPU (`DEVICE`), hoặc giảm `TARGET_FPS` |

---

## 7. Việc nên làm tiếp (theo thứ tự ưu tiên)

1. **Chạy với model YOLO thật trên vài video**, trong đó có video 07 (video có cú ngã đã từng làm bản cũ mất 13/17 khớp). Mở `*_overlay.mp4` kiểm tra bằng mắt rằng skeleton bám đúng người, và xác nhận sau cú ngã bộ lọc không còn làm rớt hàng loạt khớp. Trên dữ liệu giả vẫn thấy rớt vài khớp (do khung xương giả có độ dài xương không thực tế); cần xác nhận trên dữ liệu thật và chỉnh `max_bone_ratio`/`collapse_ratio` nếu cần.
2. **Đối chiếu với FallApp:** so `PoseResult` và schema CSV với `skeleton_io.py`, `app/sources.py`, `app/smoothing.py` của FallApp. Mã ở đây là phiên bản độc lập viết theo mô tả, chưa được ghép trực tiếp.
3. **So sánh có và không có lọc:** đặt `USE_FILTER = False` để xuất CSV thô, đối chiếu với bản lọc để chắc chắn bộ lọc không làm méo tín hiệu ngã.
4. **Đo tốc độ thực tế** trên máy đích (webcam ~30 giây, xem số dòng CSV và độ đều của `t`) để biết có chạy kịp 10 fps thời gian thực không.
5. **Cài đặt bộ phát hiện ngã** (mục 5) và viết kiểm thử đơn vị cho `SkeletonFilter` với các chuỗi tổng hợp (đứng yên, đi ngang, ngã, mất khớp, mất người).
6. **Kiểm thử webcam thật:** `StreamSource` mới chỉ được thử bằng camera giả lập 30 fps (ra đúng ~10 fps, `t` cách nhau ~0.1 giây).
