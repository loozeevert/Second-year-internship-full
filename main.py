import argparse
from pathlib import Path
from src.models.yolo import train_yolo
from src.models.faster_rcnn import train_faster_rcnn
from src.models.ssd import train_ssd
from src.models.efficientdet import train_efficientdet
from src.models.detr import train_detr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="yolo",
        choices=["yolo", "faster_rcnn", "ssd", "efficientdet", "detr"],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    train_img = "data/raw/yolo_format/images/train"
    train_lbl = "data/raw/yolo_format/labels/train"

    if args.model == "yolo":
        yaml_path = Path("data/raw/yolo_format/dataset.yaml").resolve()
        train_yolo(data_yaml=str(yaml_path), epochs=args.epochs)
    elif args.model == "faster_rcnn":
        train_faster_rcnn(
            train_img_dir=train_img,
            train_lbl_dir=train_lbl,
            epochs=args.epochs,
            batch_size=args.batch,
        )
    elif args.model == "ssd":
        train_ssd(
            train_img_dir=train_img,
            train_lbl_dir=train_lbl,
            epochs=args.epochs,
            batch_size=8,
        )
    elif args.model == "efficientdet":
        train_efficientdet(
            train_img_dir=train_img,
            train_lbl_dir=train_lbl,
            epochs=args.epochs,
            batch_size=4,
        )
    elif args.model == "detr":
        train_detr(
            train_img_dir=train_img,
            train_lbl_dir=train_lbl,
            epochs=args.epochs,
            batch_size=args.batch,
        )


if __name__ == "__main__":
    main()