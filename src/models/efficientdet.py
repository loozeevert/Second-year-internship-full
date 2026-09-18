import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

torch.set_num_threads(4)


class EfficientDetTrafficDataset(Dataset):
    def __init__(self, img_dir: str, lbl_dir: str, img_size: int = 512):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
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
                    img_resized = img.resize((self.img_size, self.img_size))
                break
            except Exception:
                idx = (idx + 1) % len(self.valid_samples)

        boxes = []
        labels = []

        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])

                    xmin = max(0.0, (xc - bw / 2.0) * self.img_size)
                    ymin = max(0.0, (yc - bh / 2.0) * self.img_size)
                    xmax = min(float(self.img_size), (xc + bw / 2.0) * self.img_size)
                    ymax = min(float(self.img_size), (yc + bh / 2.0) * self.img_size)

                    if (xmax - xmin) > 2.0 and (ymax - ymin) > 2.0:
                        boxes.append([ymin, xmin, ymax, xmax])
                        labels.append(cls_id + 1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        import torchvision.transforms.functional as TF
        img_tensor = TF.to_tensor(img_resized)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        target = {
            "bbox": boxes,
            "cls": labels,
            "img_size": torch.tensor([self.img_size, self.img_size]),
            "img_scale": torch.tensor([1.0])
        }

        return img_tensor, target


def collate_fn(batch):
    batch = [item for item in batch if len(item[1]["bbox"]) > 0]
    if len(batch) == 0:
        return None

    images = torch.stack([item[0] for item in batch])
    
    max_len = max(len(item[1]["bbox"]) for item in batch)
    
    batch_boxes = torch.zeros((len(batch), max_len, 4), dtype=torch.float32)
    batch_cls = torch.zeros((len(batch), max_len), dtype=torch.int64)
    
    for i, item in enumerate(batch):
        b = item[1]["bbox"]
        c = item[1]["cls"]
        if len(b) > 0:
            batch_boxes[i, :len(b)] = b
            batch_cls[i, :len(c)] = c

    targets = {
        "bbox": batch_boxes,
        "cls": batch_cls,
        "img_size": torch.stack([item[1]["img_size"] for item in batch]),
        "img_scale": torch.stack([item[1]["img_scale"] for item in batch])
    }

    return images, targets


def build_efficientdet(num_classes: int = 4):
    config = get_efficientdet_config("tf_efficientdet_d0")
    config.num_classes = num_classes
    config.image_size = (512, 512)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )

    model = DetBenchTrain(net, config)
    return model


def train_efficientdet(
    train_img_dir: str,
    train_lbl_dir: str,
    epochs: int = 4,
    batch_size: int = 4,
    lr: float = 0.0005,
    max_batches_per_epoch: int = 120,
):
    device = torch.device("cpu")
    print(f"EfficientDet-D0: запуск обучения на {device} ({epochs} эпох)...")

    dataset = EfficientDetTrafficDataset(train_img_dir, train_lbl_dir, img_size=512)
    print(f"Доступно обучающих изображений: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = build_efficientdet(num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        steps = 0
        total_steps = min(len(loader), max_batches_per_epoch)
        pbar = tqdm(total=total_steps, desc=f"Эпоха [{epoch+1}/{epochs}]", unit="batch", dynamic_ncols=True)

        for batch in loader:
            if batch is None:
                continue
            images, targets = batch
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = loss_dict["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            steps += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if steps >= max_batches_per_epoch:
                break

        pbar.close()
        avg_loss = running_loss / max(1, steps)
        print(f"--- Эпоха {epoch+1}/{epochs} завершена. Средний Loss: {avg_loss:.4f} ---")

    save_dir = Path("results/efficientdet")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "efficientdet_d0_traffic.pth"
    torch.save(model.model.state_dict(), save_path)
    print(f"Веса EfficientDet успешно сохранены в {save_path}")