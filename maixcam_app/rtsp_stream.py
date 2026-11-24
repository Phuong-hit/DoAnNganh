"""Dự phòng: MaixCAM chỉ phát RTSP (không chạy pose). PC sẽ tự chạy YOLO-pose.
Trên web PC nhập nguồn: rtsp://<IP>:8554/live  (địa chỉ đúng in ra log).
Chạy bằng MaixVision: mở file này rồi bấm Run."""
from maix import app, camera, image, rtsp, time

cam = camera.Camera(640, 480, image.Format.FMT_YVU420SP)   # định dạng bộ mã hóa H.264 cần
server = rtsp.Rtsp()
server.bind_camera(cam)
server.start()
print("RTSP:", server.get_url())

while not app.need_exit():
    time.sleep(1)
