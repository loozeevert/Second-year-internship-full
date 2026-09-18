from ultralytics import YOLO


def train_yolo(data_yaml: str, epochs: int = 5, batch: int = 8, imgsz: int = 640):
    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project="results",
        name="yolo_baseline",
        plots=True,
    )
    return model.val()