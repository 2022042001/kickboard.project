from ultralytics import YOLO

# 사전 학습된 YOLOv5 모델 불러오기
model = YOLO('yolov5s.pt')  # 처음이면 'yolov5n.pt'도 추천 (가벼움)

# 학습 실행
model.train(
    data='/home/a/PycharmProjects/gradu/yolov11/datasets/kick.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='kick_train'
)
