import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data config: everyday objects + people + animals
# --------------------------------------------------------------------------- #
DATA_ROOT = "./data"
IMG_SIZE = 32  # CIFAR images are natively 32x32; set 64 to upsample (slower, no new detail)

# CIFAR-100 stats (CIFAR-10 is close enough to share them)
MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)

# Max images per class. Merged "person" and the CIFAR-10 animals have far more
# images than the other classes, so capping keeps the dataset reasonably balanced.
# Set to None to disable.
TRAIN_CAP_PER_CLASS = 1000
TEST_CAP_PER_CLASS = 100

# CIFAR-100 people classes -> merged into a single "person" class
PEOPLE = {"baby", "boy", "girl", "man", "woman"}

# Everyday objects (CIFAR-100)
OBJECTS = [
    "clock", "keyboard", "lamp", "telephone", "television",
    "bed", "chair", "couch", "table", "wardrobe",
    "bottle", "bowl", "can", "cup", "plate",
    "bicycle", "bus", "motorcycle", "pickup_truck", "train",
]

# Animals from CIFAR-100
ANIMALS_C100 = [
    "bear", "camel", "cattle", "chimpanzee", "elephant", "fox", "hamster",
    "kangaroo", "leopard", "lion", "rabbit", "squirrel", "tiger", "wolf",
]

# CIFAR-100 has no cat/dog/horse/bird, so those come from CIFAR-10
ANIMALS_C10 = ["bird", "cat", "deer", "dog", "frog", "horse"]

CLASS_NAMES = OBJECTS + ANIMALS_C100 + ANIMALS_C10 + ["person"]
NUM_CLASSES = len(CLASS_NAMES)  # 41
assert len(set(CLASS_NAMES)) == NUM_CLASSES, "duplicate class names"


class RemappedDataset(Dataset):
    """
    Wraps one or more raw CIFAR datasets (loaded with no transform) and exposes
    only the selected samples with unified labels. The transform is applied here,
    so the same sample list can be reused with train/eval transforms.
    """

    def __init__(self, sources, entries, transform=None):
        self.sources = sources    # list of CIFAR datasets
        self.entries = entries    # list of (source_id, index_in_source, new_label)
        self.transform = transform

    def with_transform(self, transform):
        return RemappedDataset(self.sources, self.entries, transform)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        src, idx, label = self.entries[i]
        img, _ = self.sources[src][idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def labels(self):
        return [e[2] for e in self.entries]


def build_dataset(train: bool, cap=None, seed: int = 42):
    c100 = datasets.CIFAR100(DATA_ROOT, train=train, download=True)
    c10 = datasets.CIFAR10(DATA_ROOT, train=train, download=True)
    name_to_label = {n: i for i, n in enumerate(CLASS_NAMES)}

    wanted = {0: set(OBJECTS) | set(ANIMALS_C100), 1: set(ANIMALS_C10)}
    by_class = defaultdict(list)
    for src_id, ds in enumerate((c100, c10)):
        for idx, y in enumerate(ds.targets):
            name = ds.classes[y]
            if src_id == 0 and name in PEOPLE:
                name = "person"
            elif name not in wanted[src_id]:
                continue
            by_class[name_to_label[name]].append((src_id, idx))

    rng = random.Random(seed)
    entries = []
    for label in sorted(by_class):
        items = by_class[label]
        rng.shuffle(items)
        if cap:
            items = items[:cap]
        entries.extend((s, i, label) for s, i in items)

    return RemappedDataset([c100, c10], entries)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """Conv -> BN -> Activation, optionally downsampling with stride=2."""

    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    """Two conv layers with a skip connection. Downsamples if stride != 1."""

    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.block1 = ConvBlock(in_ch, out_ch, stride=stride, dropout=dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block1(x)
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class RobustCNN(nn.Module):
    """
    A ResNet-style CNN sized for small/medium images (e.g. 32x32 or 64x64).

    Stages progressively downsample spatial resolution while increasing
    channel depth, followed by global average pooling and a linear head.
    """

    def __init__(self, in_channels=3, num_classes=10, base_width=64,
                 blocks_per_stage=(2, 2, 2), dropout=0.1, head_dropout=0.3):
        super().__init__()

        self.stem = ConvBlock(in_channels, base_width, stride=1)

        widths = [base_width * (2 ** i) for i in range(len(blocks_per_stage))]
        stages = []
        in_ch = base_width
        for stage_idx, (width, n_blocks) in enumerate(zip(widths, blocks_per_stage)):
            for block_idx in range(n_blocks):
                # Downsample at the start of every stage except the first
                stride = 2 if (block_idx == 0 and stage_idx != 0) else 1
                stages.append(ResidualBlock(in_ch, width, stride=stride, dropout=dropout))
                in_ch = width
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(in_ch, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return self.head(x)


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="training", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=running_loss / total, acc=correct / total)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    model_config = {
        "in_channels": 3,
        "num_classes": NUM_CLASSES,
        "base_width": 64,
        "blocks_per_stage": (2, 2, 2),
        "dropout": 0.1,
        "head_dropout": 0.3,
    }
    model = RobustCNN(**model_config).to(device)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # ---- Data: CIFAR-100 (objects, people, animals) + CIFAR-10 (cat, dog, horse, ...) ----
    full_train = build_dataset(train=True, cap=TRAIN_CAP_PER_CLASS)
    test_raw = build_dataset(train=False, cap=TEST_CAP_PER_CLASS)

    full_train_aug = full_train.with_transform(train_transform)
    full_train_eval = full_train.with_transform(eval_transform)
    test_dataset = test_raw.with_transform(eval_transform)

    # Train/val split (val uses eval transforms, no augmentation)
    val_size = int(0.15 * len(full_train))
    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(full_train), generator=generator).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_dataset = Subset(full_train_aug, train_idx)
    val_dataset = Subset(full_train_eval, val_idx)

    print(f"Classes: {NUM_CLASSES} | train: {len(train_dataset)} | "
          f"val: {len(val_dataset)} | test: {len(test_dataset)}")
    print("Class names:", CLASS_NAMES)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Class-weighted loss for training (compensates for the remaining imbalance,
    # e.g. "person" and CIFAR-10 animals having more images than other classes).
    # Eval uses a plain loss so val/test numbers are comparable.
    labels = np.array(full_train.labels)
    counts = np.bincount(labels[train_idx], minlength=NUM_CLASSES)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion_train = nn.CrossEntropyLoss(weight=class_weights)
    criterion_eval = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    num_epochs = 30
    best_val_loss = float("inf")
    patience, patience_counter = 5, 0

    model_path = "best_model.pth"

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion_train, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion_eval, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "model_state": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "image_size": IMG_SIZE,
                "model_config": model_config,
                "normalization": {"mean": MEAN, "std": STD},
                "val_acc": val_acc,
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion_eval, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
