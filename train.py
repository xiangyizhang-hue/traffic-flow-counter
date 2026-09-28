import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or validate an Ultralytics YOLO model")
    parser.add_argument("data", help="dataset YAML path")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.validate_only:
        model.val(data=args.data, imgsz=args.imgsz)
    else:
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
