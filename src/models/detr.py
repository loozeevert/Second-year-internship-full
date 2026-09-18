import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection

torch.set_num_threads(4)


class DETRTrafficDataset(Dataset):
    def __init__(self, img_dir: str, lbl_dir: str, processor, img_size: int = 512):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.processor = processor
        self.img_size = img_size

        all_imgs = sorted(list(self.img_dir.glob("*.jpg")))
        self.valid_samples = []
        for img_path in all_imgs:
            lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                self.valid_samples.append((img_path, lbl_path))

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        while True:
            img_path, lbl_path = self.valid_samples[idx]
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize((self.img_size, self.img_size))
                break
            except Exception:
                idx = (idx + 1) % len(self.valid_samples)

        boxes = []
        class_labels = []

        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    xmin = max(0.0, (xc - bw / 2.0) * self.img_size)
                    ymin = max(0.0, (yc - bh / 2.0) * self.img_size)
                    box_w = bw * self.img_size
                    box_h = bh * self.img_size

                    if box_w > 2 and box_h > 2:
                        boxes.append([xmin, ymin, box_w, box_h])
                        class_labels.append(cls_id)

        target = {
            "image_id": idx,
            "annotations": [
                {"bbox": box, "category_id": label, "area": box[2] * box[3], "iscrowd": 0}
                for box, label in zip(boxes, class_labels)
            ],
        }

        encoding = self.processor(
            images=img,
            annotations=target,
            return_tensors="pt",
            do_resize=False
        )
        pixel_values = encoding["pixel_values"].squeeze(0)
        target_dict = encoding["labels"][0]

        return pixel_values, target_dict


def collate_fn_detr(batch):
    batch = [item for item in batch if item is not None and len(item[1]["class_labels"]) > 0]
    if len(batch) == 0:
        return None

    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return pixel_values, labels


def train_detr(
    train_img_dir: str,
    train_lbl_dir: str,
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
    max_batches_per_epoch: int = 180,
):
    device = torch.device("cpu")
    print(f"DETR (Vision Transformer): запуск обучения на {device} ({epochs} эпох, {max_batches_per_epoch} батчей/эпоха)...")

    model_name = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(model_name)

    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=3,
        ignore_mismatched_sizes=True,
    ).to(device)

    for param in model.model.backbone.parameters():
        param.requires_grad = False

    for param in model.model.decoder.parameters():
        param.requires_grad = True
    for param in model.class_labels_classifier.parameters():
        param.requires_grad = True
    for param in model.bbox_predictor.parameters():
        param.requires_grad = True

    dataset = DETRTrafficDataset(train_img_dir, train_lbl_dir, processor, img_size=512)
    print(f"Обучающая выборка: {len(dataset)} изображений")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_detr,
        num_workers=0,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        steps = 0
        total_steps = min(len(loader), max_batches_per_epoch)
        pbar = tqdm(total=total_steps, desc=f"DETR Эпоха [{epoch+1}/{epochs}]", unit="batch", dynamic_ncols=True)

        for batch in loader:
            if batch is None:
                continue
            pixel_values, labels = batch
            pixel_values = pixel_values.to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
            optimizer.step()

            running_loss += loss.item()
            steps += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if steps >= max_batches_per_epoch:
                break

        pbar.close()
        avg_loss = running_loss / max(1, steps)
        print(f"--- DETR Эпоха {epoch+1}/{epochs} завершена. Средний Loss: {avg_loss:.4f} ---")

    save_dir = Path("results/detr")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "detr_traffic.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Веса DETR сохранены в: {save_path}")