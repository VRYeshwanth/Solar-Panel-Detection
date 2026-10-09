from ultralytics import YOLO

def main():
    MODEL_NAME = "yolo11n.pt"
    DATASET_CONFIG = "./dataset/data.yaml"

    EPOCHS = 100
    IMAGE_SIZE = 1080
    BATCH_SIZE = 16
    WORKERS = 4

    EXPERIMENT_NAME = "baseline_100ep_1080"

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project="./runs/detect",
        name=EXPERIMENT_NAME
    )


if __name__ == "__main__":
    main()