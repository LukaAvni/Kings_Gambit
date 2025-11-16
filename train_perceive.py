from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")
    model.train(
        data="vision/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        name="train_perceive",
        patience=12,
        augment=True
    )

if __name__ == "__main__":
    main()