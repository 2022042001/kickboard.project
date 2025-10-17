from flask import Flask, Response
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)
model = YOLO("/home/a/runs/detect/train9/weights/best.pt")  # 너의 학습된 모델 경로
cap = cv2.VideoCapture(2)  # 사용 가능한 카메라 인덱스로 변경

# ROI 주차구역 사각형 예시 (직접 좌표 조정 가능)
# 예: 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래
parking_zone = np.array([[100, 200], [400, 200], [400, 400], [100, 400]])

def generate():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        boxes = results[0].boxes
        annotated_frame = results[0].plot()

        kick_count = 0
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label == 'kick':
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                point = (center_x, center_y)

                # 킥보드가 주차구역 안에 있는지 확인
                inside = cv2.pointPolygonTest(parking_zone, point, False)
                color = (0, 0, 255) if inside >= 0 else (255, 255, 255)

                kick_count += 1
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(annotated_frame, point, 5, color, -1)
                cv2.putText(annotated_frame, 'kick', (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ROI(주차구역) 시각화
        cv2.polylines(annotated_frame, [parking_zone], isClosed=True, color=(0, 255, 255), thickness=2)

        # 하단 정보 텍스트 추가
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, f'Kickboards: {kick_count}', (10, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, now, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 프레임 JPEG 인코딩 후 스트리밍
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h2>YOLO 킥보드 감지 스트리밍</h2><img src='/video_feed'>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

