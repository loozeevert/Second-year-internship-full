import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDHead
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm

torch.set_num_threads(4)


class SSDTrafficDataset(Dataset):
    def __init__(self, img_dir: str, lbl_dir: str, target_size: int = 300):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.target_size = target_size

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
                    img_resized = img.resize((self.target_size, self.target_size))
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

                    xmin = max(0.0, (xc - bw / 2.0) * self.target_size)
                    ymin = max(0.0, (yc - bh / 2.0) * self.target_size)
                    xmax = min(float(self.target_size), (xc + bw / 2.0) * self.target_size)
                    ymax = min(float(self.target_size), (yc + bh / 2.0) * self.target_size)

                    if (xmax - xmin) > 2.0 and (ymax - ymin) > 2.0:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cls_id + 1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        img_tensor = torchvision.transforms.functional.to_tensor(img_resized)
        return img_tensor, target


def collate_fn(batch):
    batch = [item for item in batch if len(item[1]["boxes"]) > 0]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))


def build_ssd300(num_classes: int = 4):
    weights = SSD300_VGG16_Weights.DEFAULT
    model = ssd300_vgg16(weights=weights)

    for param in model.backbone.parameters():
        param.requires_grad = False

    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head = SSDHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes)
    return model


def train_ssd(
    train_img_dir: str,
    train_lbl_dir: str,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 0.001,
):
    device = torch.device("cpu")
    print(f"SSD300: запуск полного обучения на {device} (300x300, все изображения, {epochs} эпох)...")

    dataset = SSDTrafficDataset(train_img_dir, train_lbl_dir, target_size=300)
    print(f"Доступно обучающих изображений: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = build_ssd300(num_classes=4).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=0.0005)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        steps = 0
        total_steps = len(loader)
        pbar = tqdm(total=total_steps, desc=f"Эпоха [{epoch+1}/{epochs}]", unit="batch", dynamic_ncols=True)

        for batch in loader:
            if batch is None:
                continue
            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            steps += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        pbar.close()
        avg_loss = running_loss / max(1, steps)
        print(f"--- Эпоха {epoch+1}/{epochs} завершена. Средний Loss: {avg_loss:.4f} ---")

    save_dir = Path("results/ssd")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "ssd300_traffic.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Веса SSD300 успешно сохранены в {save_path}")