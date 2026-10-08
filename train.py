from ultralytics import YOLO

def main():
    MODEL_NAME = "yolo11n.pt"
    DATASET_CONFIG = "./dataset/data.yaml"

    EPOCHS = 50
    IMAGE_SIZE = 640
    BATCH_SIZE = 16
    WORKERS = 4

    EXPERIMENT_NAME = "baseline_50ep_640"

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