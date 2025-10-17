import cv2
from ultralytics import YOLO
from datetime import datetime

# 모델 로딩
model = YOLO("/home/a/runs/detect/train9/weights/best.pt")  # 너가 학습시킨 모델 경로

# 웹캠 열기 (카메라 번호는 0, 1, 2 중 작동하는 걸로 선택)
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 추론
    results = model(frame)
    boxes = results[0].boxes

    # 킥보드 클래스 인식 수 세기
    kick_count = 0
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        if label == 'kick':  # 'kick'은 클래스 이름! 필요시 너가 라벨링한 이름으로 변경
            kick_count += 1

    # 바운딩 박스가 포함된 프레임 얻기
    annotated_frame = results[0].plot()

    # 킥보드 개수 왼쪽 하단에 출력
    cv2.putText(annotated_frame, f'Kickboards: {kick_count}', (10, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 오른쪽 상단에 날짜/시간 출력
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text_size, _ = cv2.getTextSize(now, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    cv2.putText(annotated_frame, now, (text_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 결과 화면 보여주기
    cv2.imshow("TWO-YOUNG Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
