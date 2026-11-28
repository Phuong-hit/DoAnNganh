# maixcam_app — chạy trên MaixCAM / MaixCAM Lite

MaixCAM chạy **cả pipeline**: camera → YOLO11-pose (NPU) → SkeletonFilter → Shift-GCN (numpy, CPU) → hậu xử lý nhánh final → cảnh báo NGÃ,
đồng thời phát ảnh + keypoint + kết quả ngã qua HTTP cho web trên PC (xem lại, lưu clip).
Phát hiện ngã (LSTM / Shift-GCN), lưu 5 cảnh báo gần nhất và giao diện xem lại đều chạy trên PC (`run_web.py`).

```
MaixCAM (main.py, cổng 8000) ──Wi-Fi──► PC: python run_web.py --host 0.0.0.0 --monitor maix://<IP MaixCAM>
                                              └─► điện thoại / laptop cùng Wi-Fi: http://<IP PC>:5000/monitor
```

| File | Vai trò |
|---|---|
| `main.py` | **Chạy file này.** Pose (NPU) + phát hiện ngã + HTTP (`/`, `/stream`, `/info`, `/bench`) |
| `src/fall_detect.py`, `src/shiftgcn_np.py` | Phát hiện ngã trên board (numpy), giống hệt web — kiểm tra: `python tests/test_maixcam_fall.py` |
| `model_classify/*.npz` | Trọng số Shift-GCN cho board. Tạo/cập nhật trên PC: `python export_maixcam_model.py` |
| `pose_preview.py` | Bản gốc của nhánh `maixcam`: chỉ vẽ khung xương lên màn hình, không phát mạng |
| `rtsp_stream.py` | Dự phòng: chỉ phát RTSP, PC tự chạy YOLO (`rtsp://<IP>:8554/live`) |
| `src/` | Code pose cho MaixPy (từ nhánh `maixcam`) |
| `model_classify/yolo11n_pose_f11_best_v5.cvimodel` + `.mud` | Model pose đã chuyển cho NPU |
| `app.yaml` | Để MaixVision cài thành app trên máy |

## Lần đầu

1. **Flash hệ điều hành** vào thẻ nhớ (MaixCAM khởi động từ thẻ): tải image MaixCAM mới nhất từ trang phát hành MaixPy, ghi bằng balenaEtcher.
2. **Cài MaixVision** trên PC, cắm MaixCAM bằng cáp USB-C → MaixVision tự tìm thấy thiết bị (card mạng USB; Windows không nhận thì cài driver RNDIS theo wiki Sipeed).
3. **Wi-Fi**: bản Lite không có màn hình → cài Wi-Fi qua MaixVision (hoặc SSH `root`/`root`). MaixCAM và PC phải **cùng một mạng**. Nên đặt IP tĩnh cho MaixCAM trong router.

## Chạy thử bằng MaixVision

1. MaixVision → mở **thư mục** `maixcam_app` (Open Project) → mở `main.py` → **Run**.
   MaixVision gửi cả thư mục (code + model) lên máy.
2. Log in ra: `Stream: http://192.168.x.y:8000/   |   web PC: maix://192.168.x.y` (IP cũng hiện trên ảnh preview).
3. Mở `http://192.168.x.y:8000/` bằng trình duyệt trên PC → thấy hình là camera OK.

## Nối với web trên PC

```bash
python run_web.py web --host 0.0.0.0 --monitor maix://192.168.x.y
```
Mở `http://127.0.0.1:5000/monitor` (hoặc `http://<IP PC>:5000/monitor` trên điện thoại).
Không dùng `--monitor` thì nhập `maix://192.168.x.y` vào ô nguồn trên trang và bấm “Bắt đầu giám sát”.
Ô nguồn ở trang chính (`/`) cũng nhận `maix://...` như nhập `0` cho webcam.

## Đo hiệu năng trên board

1. Run Project như trên, **tắt web / trình duyệt** đang xem luồng (có người xem thì board phải nén JPEG).
2. Trên PC: `python bench_maix.py --ip <IP MaixCAM>` (đo 120 s, đứng / đi lại trước camera).
   Thêm cột PC: `python bench_report.py --cpu --label cpu_shiftgcn` rồi `--compare "output/bench/report_cpu_shiftgcn_*.json"`.
3. Xem nhanh: `http://<IP>:8000/bench` (đo lại từ đầu: `/bench/reset`); log MaixVision in dòng `[bench]` mỗi 10 s.

## Tự chạy khi cắm điện

MaixVision → **Install App** (dùng `app.yaml`) → trên máy chọn app *Fall Monitor* làm app khởi động
(Settings → Auto start, hoặc theo hướng dẫn “auto start” trên wiki MaixPy cho đúng bản firmware).

## Chỉnh

Đầu `main.py`: `PORT`, `JPEG_QUALITY` (giảm nếu Wi-Fi yếu / luồng giật), `CONF_TH`, `KP_CONF_TH`, `SHOW_ON_SCREEN`.

## Chuyển lại model pose của nhóm sang .cvimodel

**Cách nhanh (đã chuẩn bị sẵn):** trong WSL Ubuntu chạy
`bash /mnt/d/documentss/DoAnNganh/FallApp/fall_web/maixcam_app/convert_pose.sh`
(tự cài uv + Python 3.10 + tpu_mlir, chuyển ONNX 320×224 đã chuẩn bị ở `output/maix_convert/`, chép `.cvimodel` mới vào `model_classify/`).

Lưu ý tên output với ultralytics hiện tại: tensor keypoint (1×51×N) là `/model.23/Concat_2_output_0`,
còn `/model.23/Concat_output_0` là tensor box 64 kênh. ONNX trong `output/maix_convert/*_maix.onnx` đã được
đổi tên để keypoint mang tên `/model.23/Concat_output_0` như MaixPy cần. Các bước thủ công bên dưới để tham khảo.

`yolo11n_pose_f11_best_v5.cvimodel` hiện được chuyển **nguyên đồ thị** (một output `output0`) nên `nn.YOLO11`
báo `get tensor idx error`. MaixPy cần model **cắt ở 3 output của head** để nó tự giải mã.
Trong lúc chờ, `main.py` tự dùng model có sẵn `/root/models/yolo11n_pose.mud` (log ghi “Model đang dùng”).

1. Xuất ONNX đúng kích thước đầu vào của MaixCAM (320×320):
   ```python
   from ultralytics import YOLO
   YOLO("yolo11n_pose_f11_best_v6.pt").export(format="onnx", imgsz=320, dynamic=False, simplify=True, opset=17)
   ```
2. Mở file `.onnx` bằng **Netron**, tìm 3 node cuối của `/model.23` (đầu vào 320 → 2100 anchor):
   | Output | Tên thường gặp | Shape |
   |---|---|---|
   | box (DFL) | `/model.23/dfl/conv/Conv_output_0` | 1×1×4×2100 |
   | điểm tin cậy | `/model.23/Sigmoid_output_0` | 1×1×2100 |
   | keypoint (Concat của cv4) | `/model.23/Concat_output_0` | 1×51×2100 |
   Tên có thể lệch theo bản ultralytics, **lấy đúng tên trong Netron theo shape ở trên**.
3. Chuyển bằng tpu-mlir (docker của Sophgo), theo hướng dẫn “chuyển YOLO11 cho MaixCAM” trên wiki MaixPy:
   ```bash
   model_transform.py --model_name yolo11n_pose_f11 --model_def best.onnx --input_shapes [[1,3,320,320]] \
     --mean 0,0,0 --scale 0.00392156862745098,0.00392156862745098,0.00392156862745098 \
     --keep_aspect_ratio --pixel_format rgb \
     --output_names "/model.23/dfl/conv/Conv_output_0,/model.23/Sigmoid_output_0,/model.23/Concat_output_0" \
     --test_input test.jpg --test_result top_outputs.npz --mlir model.mlir
   run_calibration.py model.mlir --dataset ./calib_images --input_num 200 -o cali_table
   model_deploy.py --mlir model.mlir --quantize INT8 --calibration_table cali_table --processor cv181x \
     --test_input test.jpg --test_reference top_outputs.npz --model yolo11n_pose_f11_best_v5.cvimodel
   ```
   `calib_images`: ~200 ảnh giống bối cảnh thật (trong phòng, có người đứng / ngồi / nằm).
4. Chép `.cvimodel` mới đè vào `model_classify/`, giữ file `.mud` hiện tại (đã có phần `[extra]`), Run Project lại.

## Lỗi thường gặp

- **Web báo “Không kết nối được MaixCAM”**: sai IP, khác mạng Wi-Fi, hoặc `main.py` chưa chạy. Thử mở `http://<IP>:8000/` trên trình duyệt trước.
- **“mất kết nối, đang thử lại…”**: Wi-Fi chập chờn — web tự kết nối lại, không cần làm gì.
- **API MaixPy đổi giữa các bản firmware**: nếu `img.to_jpeg()` lỗi, `main.py` tự chuyển sang OpenCV; lỗi khác thì đối chiếu ví dụ trong thư mục `examples` của MaixPy đúng bản đã flash.
