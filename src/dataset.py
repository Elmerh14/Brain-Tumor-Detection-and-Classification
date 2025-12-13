# src/dataset.py
from pathlib import Path
from typing import Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"] # Here change the names 

def find_data_root() -> Path:
    """
    Kaggle mounts datasets under /kaggle/input/<dataset-slug>/...
    We'll search there first, otherwise fallback to ./data
    """
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # try to find a folder containing "Training" and "Testing"
        for p in kaggle_input.rglob("*"):
            if p.is_dir():
                training = p / "Training"
                testing  = p / "Testing"
                if training.exists() and testing.exists():
                    return p
    # local fallback
    return Path("data")

def build_transforms(img_size: int = 224, train: bool = True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])

class FolderImageDataset(Dataset):
    def __init__(self, root_dir: Path, split: str, img_size: int = 224, train: bool = True):
        """
        root_dir should contain Training/ and Testing/
        split: "Training" or "Testing"
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = build_transforms(img_size=img_size, train=train)

        self.samples: List[Tuple[Path, int]] = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root_dir / split / class_name 
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing folder: {class_dir}")

            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root_dir / split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
