from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)

# YOLO 모델 로드
model = YOLO("/home/a/PycharmProjects/gradu/yolov5/runs/detect/kick_train/weights/best.pt")

# 카메라 열기
cap = cv2.VideoCapture(2)

# 수동 ROI: 마우스로 지정한 주차공간 좌표
PARKING_ROI = (140, 220, 300, 220)  # (x, y, w, h)

def is_bottom_center_inside_roi(box, roi):
    x, y, w, h = roi
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int(y2)  # 바운딩 박스의 '하단 중앙'
    return x <= cx <= x + w and y <= cy <= y + h

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        annotated_frame = results.plot()

        # ROI 표시 (민트색: #31B4B3 → BGR = (179, 180, 49))
        x, y, w, h = PARKING_ROI
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (179, 180, 49), 2)
        cv2.putText(annotated_frame, "PARKING ZONE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (179, 180, 49), 2)

        # YOLO 결과 처리
        for box in results.boxes.xyxy.cpu().numpy():
            if is_bottom_center_inside_roi(box, PARKING_ROI):
                # 정상 주차: 초록색 박스만
                cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 255, 0), 2)
            else:
                # 비정상 주차: 빨간색 박스만
                cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 0, 255), 2)

        # 시간 표시
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, now, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
