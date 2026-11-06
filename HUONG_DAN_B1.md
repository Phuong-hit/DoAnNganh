# Hướng dẫn B1: huấn luyện LSTM phát hiện ngã trên VSCode

Tài liệu này hướng dẫn từng bước làm **B1 (phân loại ngã bằng LSTM)** trên máy cá nhân bằng VSCode,
không cần Google Colab, rồi so sánh với ST-GCN (`best_stgcn.pt` của code_1) trên **cùng một tập test**.

```
 data/dataset.npz ──► train.py --arch lstm ──► models/best_lstm.pt ──► evaluate.py (so với best_stgcn.pt)
 (cửa sổ 2 s, đã                              (+ .history.csv,              │
  chuẩn hóa, chia                               .report.json)               ▼
  theo người)                                                   main.py / web: chạy trên video, webcam
```

---

## 1. Có cần Google Colab không?

**Không.** Colab chỉ cho GPU miễn phí. Dataset của nhóm rất nhỏ:

| Tập | Cửa sổ | Ngã | Không ngã | Người |
|---|---|---|---|---|
| train | 1089 | 245 | 844 | Subject 2, Subject 4 |
| val | 272 | 91 | 181 | Subject 1 |
| test | 467 | 95 | 372 | Subject 3 |

Mỗi cửa sổ chỉ là 20 khung × 17 khớp × 3 số. LSTM (~0,3 triệu tham số) chạy CPU laptop vài phút; ST-GCN
(~3 triệu tham số) chậm hơn nhưng vẫn chạy được trên CPU. Có GPU NVIDIA thì nhanh hơn, code tự dùng.

Checkpoint `best_stgcn.pt` lưu từ GPU của Colab vẫn nạp được trên máy chỉ có CPU (code tự chuyển về CPU).

---

## 2. Cài đặt (làm một lần)

### 2.1 Phần mềm

1. **Python 3.10, 3.11 hoặc 3.12** (64-bit) từ python.org. Khi cài trên Windows, tích **"Add python.exe to PATH"**.
   Tránh bản quá mới nếu PyTorch chưa hỗ trợ.
2. **VSCode** + extension **Python** (Microsoft). Mở project lần đầu, VSCode sẽ gợi ý cài (file `.vscode/extensions.json`).

### 2.2 Mở project và tạo môi trường ảo

1. Giải nén `fall_web`, trong VSCode chọn **File → Open Folder…** → thư mục `fall_web`
   (thư mục chứa `train.py`, `src/`, `web/`).
2. Mở terminal: **Terminal → New Terminal**, rồi chạy:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1          # lỗi "running scripts is disabled": chạy trước
                                     #   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. **Ctrl+Shift+P → "Python: Select Interpreter"** → chọn `.venv`. (Trên macOS/Linux sửa
   `python.defaultInterpreterPath` trong `.vscode/settings.json` thành `.venv/bin/python`.)

### 2.3 Cài thư viện

**Cài PyTorch trước**, chọn một trong hai:

```bash
# Chỉ có CPU (hoặc không chắc): đủ cho B1
pip install torch

# Có GPU NVIDIA: lấy lệnh đúng phiên bản CUDA tại https://pytorch.org/get-started/locally/
# ví dụ:  pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Sau đó cài phần còn lại:

```bash
pip install -r requirements.txt
```

### 2.4 Kiểm tra

```bash
python -c "import torch; print(torch.__version__, 'GPU:', torch.cuda.is_available())"
python -m unittest discover tests
```

Kết quả mong đợi: `OK`, **không còn dòng `skipped 'chưa cài torch'`**. Có 3 test quan trọng cho B1:
- `test_lstm_forward_variants`: LSTM chạy được với các tùy chọn;
- `test_train_lstm_and_evaluate`: train 3 epoch trên dữ liệu giả, lưu checkpoint, nạp lại, đánh giá;
- `test_code1_stgcn_checkpoint`: `best_stgcn.pt` của code_1 nạp được và dự đoán trên tập test.

Nếu có test lỗi, gửi lại toàn bộ thông báo lỗi để sửa trước khi train thật.

---

## 3. Dữ liệu và model có sẵn

| File | Là gì |
|---|---|
| `data/dataset.npz` | Dataset của code_1 (do `build_dataset.py` tạo, chia train/val/test theo người) |
| `models/best_stgcn.pt` | ST-GCN code_1 đã huấn luyện trên Colab — mốc để so sánh |
| `src/model_lstm.py` | Model LSTM của B1 |
| `train.py` | Huấn luyện `--arch lstm` hoặc `--arch stgcn` |
| `evaluate.py` | So sánh nhiều checkpoint trên cùng tập test |
| `main.py` | Chạy model trên video/webcam (cửa sổ OpenCV) |
| `configs/code1.yaml` | Cấu hình pose **giống lúc tạo dataset** (conf 0.25, imgsz 224×320, yolo11n-pose) |

Mỗi cửa sổ trong dataset: `X [20, 17, 3]` = 20 khung (2 giây @ 10 fps) × 17 khớp COCO × (x, y đã chuẩn hóa, mask).
Chuẩn hóa (`src/features.py`): lấy tâm hai hông làm gốc, chia cho độ dài xương trung vị, **một tâm và một tỉ lệ
cho cả cửa sổ** để giữ được chuyển động hạ thấp người khi ngã. Nhãn 1 nếu cửa sổ chồng ≥ 30% đoạn ngã.

---

## 4. Huấn luyện LSTM

### Cách 1: nút F5 trong VSCode

Tab **Run and Debug** (Ctrl+Shift+D) → chọn **"B1: Train LSTM"** → F5.

### Cách 2: terminal

```bash
python train.py --arch lstm
```

(mặc định `--data data/dataset.npz --out models/best_lstm.pt --epochs 60 --patience 15`)

### Màn hình khi chạy

```
Dataset : .../data/dataset.npz (định dạng code_1, cửa sổ 20 khung @ 10 fps)
train: 1089 cửa sổ | ngã  245 | không ngã  844 | 85 video | Subject 2, Subject 4
val  :  272 cửa sổ | ngã   91 | ...
Model   : lstm | 2xx,xxx tham số
Epoch 001/60 | loss 0.6xxx | val acc 0.7xx | P 0.6xx | R 0.8xx | F1 0.7xx | 1.2s
...
Early stopping tại epoch 3x.
===== KẾT QUẢ TEST (theo cửa sổ) =====
Ngưỡng 0.50 : Acc ... | P ... | R ... | F1 ... | Spec ... | TP= FN= FP= TN=
Ngưỡng 0.4x : ...
AUC ROC     : ...
Theo video (43 video, ngưỡng 0.4x): ...
```

(các số chỉ minh họa)

### Quá trình huấn luyện (giống train.py của code_1, để so sánh công bằng)

1. Trọng số lớp tính từ train (lớp ngã ít hơn nên được tăng trọng số).
2. AdamW (lr 1e-3, weight decay 1e-4) + CosineAnnealing.
3. Tăng cường mỗi batch: lật ngang 50% (đổi trái/phải, đổi dấu x) + nhiễu Gauss 0.01 trên khớp quan sát được.
4. Sau mỗi epoch đo F1 lớp ngã trên **val**; lưu checkpoint khi (F1, accuracy) tốt hơn; dừng sớm sau 15 epoch không cải thiện.
5. Nạp checkpoint tốt nhất → **chọn ngưỡng P(ngã) cho F1 cao nhất trên val** (lưu vào checkpoint) → lúc này mới đánh giá **test**.

Tập test **không bao giờ** được dùng để chọn epoch, ngưỡng hay siêu tham số.

### File được tạo

| File | Nội dung |
|---|---|
| `models/best_lstm.pt` | Checkpoint: `arch="lstm"`, trọng số, `model_kwargs`, `window_size`, `fps`, `threshold`, `val_f1`… |
| `models/best_lstm.history.csv` | loss, val acc/P/R/F1, thời gian từng epoch → vẽ đường học trong báo cáo |
| `models/best_lstm.report.json` | Toàn bộ kết quả val/test, ngưỡng, số tham số, thời gian train, tham số dòng lệnh |

---

## 5. So sánh ST-GCN và LSTM

Có ba cách, cùng một cách tính (cùng tập test, cùng các cửa sổ):

| Cách | Lệnh | Dùng khi |
|---|---|---|
| **Trang "So sánh model" trên web** | `python run_web.py --config configs/code1.yaml` → mở http://127.0.0.1:5000/compare (hoặc bấm **So sánh model →** ở đầu trang chính) | Xem trực quan, bấm vào từng video |
| **Báo cáo HTML một file** | `python evaluate.py models/best_stgcn.pt models/best_lstm.pt --html output/compare_test.html` | Đính kèm báo cáo, gửi cho thầy/nhóm (mở bằng trình duyệt, không cần server) |
| **Bảng trong terminal** | `python evaluate.py models/best_stgcn.pt models/best_lstm.pt --device cpu --csv output/compare_test.csv` | Lấy số nhanh, chép vào Excel |

(hoặc F5 với **"B1: So sánh ST-GCN và LSTM (test)"** / **"B1: Báo cáo so sánh HTML"**)

### 5.1 Trang "So sánh model"

Tích chọn `best_stgcn.pt` và `best_lstm.pt` (mặc định đã chọn sẵn và tự chạy nếu đã cài torch), chọn dataset
và tập (**test** hoặc **val**), bấm **So sánh**. Trang gồm:

| Khối | Xem gì |
|---|---|
| **Kết luận nhanh** | Model nào F1 cao nhất, chênh lệch có đáng tin không, model nào nhanh/nhẹ hơn bao nhiêu |
| **Chỉ số chính** | F1 (kèm khoảng tin cậy 95%), Recall, Precision, Specificity, Accuracy, AUC, TP/FN/FP/TN, số video ngã phát hiện được, số video sinh hoạt bị báo nhầm, số tham số, dung lượng, ms/cửa sổ. ★ = tốt nhất. Nút chuyển **ngưỡng của checkpoint** ↔ **ngưỡng 0.5** |
| **Biểu đồ chỉ số** | Cột Acc / Precision / Recall / F1 / Spec / AUC của từng model |
| **Ma trận nhầm lẫn** | Đúng/sai từng loại; ô **thật ngã – đoán không ngã (FN)** là lỗi nguy hiểm nhất |
| **Đường ROC** | So khả năng phân biệt ngã/không ngã ở mọi ngưỡng |
| **F1 theo ngưỡng** | Model nào ít nhạy với việc chọn ngưỡng; chấm tròn = ngưỡng đang dùng |
| **Khác biệt có đáng tin không** | Khoảng tin cậy 95% của chênh lệch F1 (bootstrap theo video). **Chứa 0 → chưa kết luận được** model nào tốt hơn |
| **Từng video** | Mỗi video: P(ngã) lớn nhất và kết luận của từng model (✓ phát hiện, ✗ bỏ sót, ⚠ báo nhầm). Lọc "Hai model khác nhau" để thấy ngay video mà LSTM và ST-GCN bất đồng |
| **Chi tiết video** | Bấm một dòng: P(ngã) từng cửa sổ của cả hai model, vùng đỏ là đoạn ngã thật |
| **Tải về** | CSV bảng chỉ số, JSON đầy đủ, báo cáo HTML một file, In / lưu PDF |

Kết quả CSV/JSON/HTML là của lần bấm **So sánh** gần nhất.

### 5.2 Bảng trong terminal

Ví dụ bảng (số minh họa):

```
         model   arch     params    MB  ms/cửa sổ    AUC    Acc      P      R     F1   Spec  ngưỡng  F1@ngưỡng  R video  Spec video
best_stgcn.pt  stgcn  3,0xx,xxx  12.2       x.xx  0.9xx  0.9xx  0.8xx  0.8xx  0.8xx  0.9xx     0.5      0.8xx     1.0       0.9x
 best_lstm.pt   lstm    2xx,xxx   1.x       x.xx  0.9xx  ...
```

**Đọc bảng:**
- **R (recall)**: tỉ lệ cửa sổ ngã được phát hiện. Với hệ thống cảnh báo ngã, **bỏ sót nguy hiểm hơn báo nhầm**,
  nên ưu tiên R cao, rồi mới xét P.
- **P (precision)**, **Spec (specificity)**: báo nhầm nhiều hay ít.
- **F1**: cân bằng P và R — tiêu chí chọn model chính.
- **AUC**: chất lượng xếp hạng P(ngã), không phụ thuộc ngưỡng.
- **R video / Spec video**: tính theo video (một video ngã được coi là phát hiện nếu có ít nhất một cửa sổ
  vượt ngưỡng) — gần với trải nghiệm thật hơn số theo cửa sổ.
- **params, MB, ms/cửa sổ**: chi phí chạy trên thiết bị biên. Đo bằng `--device cpu`.

Tập test chỉ có **một người (Subject 3)**, nên chênh vài phần trăm F1 có thể do ngẫu nhiên. Muốn kết luận chắc:
chạy mỗi model với 3 seed (`--seed 1`, `--seed 2`, `--seed 3`, `--out` khác nhau) và báo cáo trung bình ± độ lệch.

Muốn so sánh hoàn toàn công bằng về môi trường, có thể train lại ST-GCN trên máy bằng cấu hình
**"B1: Train ST-GCN (huấn luyện lại trên máy)"** (`--out models/stgcn_local.pt`), rồi đưa cả ba vào `evaluate.py`.

---

## 6. Thử nghiệm siêu tham số LSTM

Truyền tham số kiến trúc bằng `--model-kw` (lặp lại được). Các giá trị được lưu trong checkpoint, nên
web/`main.py`/`evaluate.py` tự dựng đúng model khi nạp.

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `hidden` | 128 | Kích thước trạng thái ẩn LSTM |
| `layers` | 2 | Số tầng LSTM |
| `embed` | 128 | Kích thước vector mỗi khung trước LSTM |
| `bidirectional` | false | LSTM hai chiều (đọc cửa sổ cả xuôi lẫn ngược) |
| `dropout` | 0.3 | Dropout |
| `velocity` | true | Thêm vận tốc (dx, dy) của từng khớp giữa hai khung liên tiếp |

Gợi ý các lần chạy (đặt `--out` khác nhau để không ghi đè):

```bash
python train.py --arch lstm --out models/lstm_base.pt
python train.py --arch lstm --model-kw velocity=false --out models/lstm_novel.pt          # vai trò của vận tốc
python train.py --arch lstm --model-kw bidirectional=true --out models/lstm_bi.pt
python train.py --arch lstm --model-kw hidden=64 --model-kw layers=1 --out models/lstm_small.pt   # nhẹ cho thiết bị biên
python train.py --arch lstm --model-kw hidden=256 --clip-grad 1.0 --out models/lstm_big.pt
python evaluate.py models/best_stgcn.pt models/lstm_*.pt --split val     # CHỌN trên val
python evaluate.py models/best_stgcn.pt models/lstm_base.pt --html output/compare_test.html   # chỉ báo cáo test cho model đã chọn
```

Tham số huấn luyện khác: `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--patience`, `--jitter`,
`--no-flip`, `--clip-grad`, `--no-tune-threshold`, `--seed`, `--device`. Xem `python train.py --help`.

**Nguyên tắc:** chọn cấu hình bằng **val**, chỉ nhìn **test** cho cấu hình cuối cùng.

---

## 7. Chạy model đã train trên video / webcam

### Bằng cửa sổ OpenCV (`main.py`)

```bash
python main.py --source "D:/dataset/Subject 3/FALL/01.mp4" --checkpoint models/best_lstm.pt
python main.py --source 0 --checkpoint models/best_lstm.pt                   # webcam
python main.py --source video.mp4 --checkpoint models/best_lstm.pt --save output/pred.mp4 --log output/pred.csv
```

Góc phải trên: **FALL 0.87** (đỏ) / **no fall 0.12** (xanh) / **FALL? ...** (đang gom đủ 20 khung đầu).
Kết thúc in các đoạn thời gian bị báo ngã. Nhấn `q` để thoát.

### Bằng giao diện web

```bash
python run_web.py --config configs/code1.yaml                    # tự chọn best_lstm.pt nếu có
python run_web.py --config configs/code1.yaml --set classifier.checkpoint=models/best_lstm.pt
```

Mở http://127.0.0.1:5000. Phía trên khung video có **khung trạng thái phát hiện ngã**:

| Phần | Ý nghĩa |
|---|---|
| Chữ lớn | **NGÃ!** (đỏ, nhấp nháy) / **KHÔNG NGÃ** (xanh) / **ĐANG GOM KHUNG x/20** (2 giây đầu chưa đủ cửa sổ) / **CHƯA BẬT PHÁT HIỆN NGÃ** (kèm lý do) |
| Số % và thanh ngang | Tỉ lệ ngã P(ngã) của lần dự đoán mới nhất; vạch đen trên thanh là ngưỡng |
| Biểu đồ | P(ngã) theo thời gian, nét đứt = ngưỡng, vùng đỏ = đoạn bị báo ngã, vạch dọc = thời điểm hiện tại |
| Dòng cuối | Các lần phát hiện ngã (thời điểm, %) |
| Ô **Model** | Đổi checkpoint ngay trên web (`best_lstm.pt`, `best_stgcn.pt`, …, hoặc tắt), không cần khởi động lại |

Dải trạng thái trên cùng cũng có mục **Ngã: NGÃ (92%)** / **không ngã (3%)**.

Khi mở web, model được chọn tự động (`classifier.checkpoint: auto`): `models/best_lstm.pt` nếu đã train, không thì
`models/best_stgcn.pt`. Nếu khung báo "chưa cài torch" thì làm lại mục 2.3.

### Vì sao dùng `configs/code1.yaml`

Model học trên skeleton do `extract_skeleton.py` tạo với **conf 0.25, kp-conf 0.25, imgsz 224×320, yolo11n-pose,
SkeletonFilter**. Lúc chạy phải tạo skeleton **giống hệt** như vậy, nếu không model gặp dữ liệu khác phân bố và
kết quả kém đi. `configs/code1.yaml` đặt đúng các giá trị này. (`configs/web.yaml` là cấu hình gốc của web, dùng
yolov8n-pose 640, conf 0.4 — chỉ hợp với model huấn luyện trên dữ liệu trích bằng cấu hình đó.)

`main.py` cũ của code_1 tự lấy pose bằng `predict()` (không lọc, không imgsz), nên đã được thay bằng bản dùng
đúng pipeline trên.

---

## 8. Tạo lại dataset khi có video mới

### Cách A: script của code_1 (cho dataset giống `data/dataset.npz`)

```bash
# 1. Trích skeleton, mỗi lần một người + một loại (subject, kind không có dấu "_")
python extract_skeleton.py "D:/dataset/Subject 5/FALL" --subject "Subject 5" --kind Fall
python extract_skeleton.py "D:/dataset/Subject 5/ADL"  --subject "Subject 5" --kind ADL
# 2. Khoảng ngã từ bảng chú thích (cột "File Name", "Classes": "Falling (SW)[3.4 to 6]; ...")
python make_intervals.py "D:/dataset/Subject 5/FALL/annotations.xlsx" --subject "Subject 5" --kind Fall
# 3. Gộp, cắt cửa sổ, chuẩn hóa, chia theo người -> output/dataset.npz
python build_dataset.py
# 4. Train trên dataset mới
python train.py --arch lstm --data output/dataset.npz --out models/lstm_v2.pt
```

Thêm người mới sẽ làm `build_dataset.py` chia lại train/val/test (chia ngẫu nhiên theo người, seed 0) —
khi đó **phải đánh giá lại cả ST-GCN** trên tập test mới để so sánh.

### Cách B: giao diện web

Quét video → điền `fall_start_s`, `fall_end_s`, `subject_id`, `split` trong `data/metadata.csv` → Trích tất cả →
**Tạo dataset** → `python train.py --arch lstm --data output/<cấu hình>/dataset/windows.npz`.
Nếu cột `split` để trống, `train.py` tự chia theo người giống `build_dataset.py`. Nhớ chạy web bằng
`--config configs/code1.yaml` để skeleton giống cách A.

---

## 9. Kiến trúc LSTM (cho báo cáo)

```
x [N, 3, 20, 17] ──► [N, 20, 17×3]  (duỗi 17 khớp × (x, y, mask) thành vector mỗi khung)
                 ──► + vận tốc [N, 20, 17×2]  (dx, dy giữa hai khung; 0 nếu khớp thiếu ở một trong hai khung)
                 ──► BatchNorm theo đặc trưng (85)
                 ──► Linear 85→128 + ReLU + Dropout 0.3          (mỗi khung -> một vector 128)
                 ──► LSTM 2 tầng, hidden 128                       (đọc tuần tự 20 khung)
                 ──► [trung bình 20 đầu ra ; đầu ra khung cuối]   (256)
                 ──► Dropout + Linear 256→2 ──► logits (không ngã, ngã)
```

So với ST-GCN:

| | ST-GCN | LSTM |
|---|---|---|
| Không gian (giữa các khớp) | Tích chập đồ thị theo khung xương (biết khớp nào nối khớp nào) | Không biết cấu trúc; một lớp Linear trộn mọi khớp |
| Thời gian | Tích chập thời gian (kernel 9 khung) | Hồi quy tuần tự qua 20 khung |
| Vận tốc | Học ngầm qua tích chập thời gian | Đưa vào trực tiếp (`velocity=True`) |
| Kích thước | ~3 triệu tham số, 12 MB | ~0,3 triệu tham số, ~1 MB |
| Đầu vào, dataset, chuẩn hóa, cửa sổ | Giống nhau | Giống nhau |

Vì đầu vào, cách chia dữ liệu, cách huấn luyện và cách chọn ngưỡng giống nhau, khác biệt kết quả phản ánh
khác biệt kiến trúc.

---

## 10. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Chưa cài PyTorch` / `No module named 'torch'` | Chưa chọn đúng `.venv` (mục 2.2 bước 3) hoặc chưa `pip install torch` |
| `No module named 'src'` | Chạy từ sai thư mục. Mở đúng thư mục `fall_web` trong VSCode; các script tự thêm gốc project vào đường dẫn |
| `Không tìm thấy dataset` | Kiểm tra `data/dataset.npz`, hoặc truyền `--data <đường dẫn>` |
| `val/test cần có cả lớp 0 và lớp 1` | Dataset mới có ít hơn 3 người hoặc một tập không có video ngã — thêm dữ liệu hoặc điền `split` tay |
| `Kiến trúc 'lstm' chưa đăng ký` | Thiếu `src/model_lstm.py`, hoặc file đó có lỗi import (xem dòng lỗi phía trên) |
| `Error(s) in loading state_dict` | Truyền `--model-kw` lúc train nhưng checkpoint cũ không có; luôn nạp qua `FallClassifier`/`evaluate.py` (tự đọc `model_kwargs`) |
| Loss = `nan` | Giảm `--lr 5e-4`, thêm `--clip-grad 1.0` |
| Val F1 dao động mạnh | Val chỉ có một người — bình thường; tăng `--patience` hoặc so nhiều seed |
| `main.py` không hiện cửa sổ | Cài `opencv-python` (không phải bản `-headless`), hoặc dùng `--no-show --save out.mp4` |
| YOLO tải `yolo11n-pose.pt` lâu / lỗi mạng | Tải sẵn file vào `models/` rồi `--set pose.model_path=models/yolo11n-pose.pt` |
| ByteTrack báo thiếu `lap` | `pip install lap` |

---

## 11. Nằm bình thường bị báo là ngã

### Nguyên nhân (đo trên `data/dataset.npz`)

Model học theo nhãn. Trong dataset hiện tại, **khoảng 22% cửa sổ có nhãn "ngã" thực chất là người đang nằm yên**
(thân nằm ngang gần suốt 2 giây, hông không đổi độ cao, gần như không cử động), trông y hệt người nằm nghỉ:

| Tập | Nhãn ngã | Trong đó là nằm yên | Video ngã bị gán nhãn ngã cho TOÀN BỘ cửa sổ |
|---|---|---|---|
| train | 245 | 54 (22%) | 10 |
| val | 91 | 21 (23%) | 5 |
| test | 95 | 21 (22%) | 2 |

Hai nguồn gây lỗi:
1. **Thiếu khoảng ngã:** `build_dataset.py` (code_1) gán nhãn 1 cho **cả video** khi video ngã không có dòng trong
   `data/fall_intervals.csv`, nên cả đoạn đứng lẫn đoạn nằm sau ngã đều thành "ngã".
2. **Khoảng ngã trong bảng chú thích kéo dài sang lúc nằm**, nên các cửa sổ nằm yên sau cú ngã cũng có nhãn 1.

Kết quả: model học "nằm ngang = ngã" thay vì "chuyển động rơi = ngã".

### Cách sửa

```bash
python check_labels.py data/dataset.npz                  # xem video / cửa sổ nào có vấn đề
python check_labels.py data/dataset.npz --fix --out data/dataset_clean.npz   # đã tạo sẵn trong gói
python train.py --arch lstm  --data data/dataset_clean.npz --out models/lstm_clean.pt
python train.py --arch stgcn --data data/dataset_clean.npz --out models/stgcn_clean.pt
```

`--fix` đổi nhãn các cửa sổ **nằm yên** từ 1 → 0 ở train và val (75 cửa sổ), giữ nguyên X. Model vẫn học cú ngã
từ các cửa sổ có chuyển động rơi. Tập test **giữ nguyên** để so được với `best_stgcn.pt` cũ; thêm `--fix-test`
nếu nhóm thống nhất "nằm yên sau ngã không tính là ngã" cho cả đánh giá (khi đó so mọi model trên file mới).

Hệ quả cần biết: sau cú ngã, khi người đã nằm yên, web sẽ chuyển về "KHÔNG NGÃ" — cảnh báo vẫn được ghi ở danh
sách "Phát hiện ngã ... lần" trên web. Đây là hành vi đúng của bộ phát hiện **sự kiện** ngã.

Sửa tận gốc (nên làm khi tạo lại dataset):
- Kiểm tra `data/fall_intervals.csv` cho các video ở mục 1 của `check_labels.py`: khoảng ngã chỉ nên bao
  **lúc rơi** (từ khi mất thăng bằng tới khi chạm đất), không bao đoạn nằm sau đó.
- `build_dataset.py` trong gói này đã đổi: video ngã **thiếu khoảng ngã sẽ bị bỏ qua** thay vì gán nhãn cả video.
- Quay thêm video sinh hoạt có **nằm xuống giường/sàn chậm, nằm nghỉ, trở mình** (nhãn ADL) để model thấy nhiều
  ví dụ "nằm nhưng không ngã".

Kiểm tra lại bằng web: chạy `main.py`/web với model mới, nằm xuống bình thường trước webcam → phải là "KHÔNG NGÃ".

---

## 12. Gộp val vào train (không dùng val)

**Được**, và hợp lý khi dữ liệu ít (thêm Subject 1 → train có 3 người, nhiều hơn 25% cửa sổ). **Test (Subject 3)
giữ nguyên, không bao giờ gộp.**

Nhưng val đang làm hai việc, gộp rồi phải thay thế:

| Val đang dùng để | Khi gộp val thì |
|---|---|
| Chọn epoch tốt nhất + early stopping | Train **đúng số epoch** đã biết từ lần chạy có val |
| Chọn ngưỡng P(ngã) | Dùng **ngưỡng cố định** đã chọn ở lần chạy có val (hoặc 0.5) |

`train.py` đã hỗ trợ sẵn — làm hai bước:

```bash
# 1. Chạy thường (có val) để biết epoch tốt nhất + ngưỡng -> models/best_lstm.report.json
python train.py --arch lstm --data data/dataset_clean.npz --out models/best_lstm.pt
# 2. Train lại trên train + val, lấy epoch + ngưỡng từ report bước 1, không early stopping
python train.py --arch lstm --data data/dataset_clean.npz --merge-val \
                --from-report models/best_lstm.report.json --out models/lstm_trainval.pt
```

(hoặc tự đặt: `--merge-val --stop-epoch 25 --threshold 0.45`). Lịch learning rate giữ giống lần chạy bước 1.
Checkpoint lưu `merged_val: true`; web, `main.py`, `evaluate.py` dùng như checkpoint thường.

**So sánh công bằng:** nếu LSTM gộp val thì ST-GCN cũng phải gộp val (làm hai bước như trên với `--arch stgcn`),
rồi so trên cùng tập test:

```bash
python evaluate.py models/stgcn_trainval.pt models/lstm_trainval.pt --data data/dataset_clean.npz --html output/compare.html
```

Lưu ý: sau khi gộp, không còn số liệu val để chọn giữa các biến thể LSTM. Hãy chọn cấu hình (mục 6) **trước**
bằng val, rồi mới gộp val để train bản cuối.

---

## 13. Nếu vẫn muốn chạy trên Colab

Không cần sửa code. Trong một ô notebook:

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/fall_web          # thư mục đã tải lên Drive
!pip install -q -r requirements.txt
!python train.py --arch lstm --device cuda
!python evaluate.py models/best_stgcn.pt models/best_lstm.pt
```

Tải `models/best_lstm.pt` về máy để chạy với web / `main.py`.
