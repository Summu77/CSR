"""
Dataset classes for CLIP adversarial robustness evaluation.

Supports:
  - Benign (clean) image loading
  - Adversarial image loading (with automatic .png suffix handling)
  - Paired clean/adv loading for training or analysis
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights


class CLIPDataset(Dataset):
    """
    General-purpose dataset for CLIP robustness evaluation.

    Reads images and labels from a CSV file. Supports both clean and 
    adversarial evaluation by pointing `img_root` to the appropriate directory.

    Args:
        csv_path: Path to labels.csv (columns: filename, label_text).
        img_root: Directory containing the images.
        dataset_name: Name of the dataset (e.g., "General/CIFAR10").
        sample_n: If set, randomly sample this many examples.
        all_classes: Override the class list (otherwise inferred from CSV).
        transform: Optional torchvision transform applied to PIL images.
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        dataset_name: str = "",
        sample_n: int = None,
        all_classes: list = None,
        transform: transforms.Compose = None,
    ):
        self.img_root = img_root
        self.dataset_name = dataset_name
        self.transform = transform

        # --- Class loading ---
        if all_classes is not None:
            self.all_classes = all_classes
        elif "ImageNet" in dataset_name:
            self.all_classes = ResNet50_Weights.DEFAULT.meta["categories"]
        else:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Label file not found: {csv_path}")
            full_df = pd.read_csv(csv_path)
            self.all_classes = sorted(list(set(full_df.iloc[:, 1].values)))

        # Build class-to-index mapping (handles comma-separated synonyms)
        self.class_to_idx = {}
        for idx, class_string in enumerate(self.all_classes):
            self.class_to_idx[class_string] = idx
            for s in class_string.split(","):
                self.class_to_idx[s.strip()] = idx

        # --- Sample loading ---
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Label file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if sample_n and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=42)

        self.image_files = df.iloc[:, 0].values
        self.labels = df.iloc[:, 1].values

    def _resolve_label(self, label_text: str) -> int:
        """Map label text to integer index with fuzzy fallback."""
        if label_text in self.class_to_idx:
            return self.class_to_idx[label_text]
        # Fuzzy matching
        lt = label_text.lower()
        for key, val in self.class_to_idx.items():
            if lt == key.lower() or lt in key.lower().split(","):
                return val
        return 0  # fallback

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        label_text = self.labels[idx]
        label_idx = self._resolve_label(label_text)

        # Resolve image path (handle .jpg -> .png suffix mismatch for adv samples)
        img_path = os.path.join(self.img_root, filename)
        if not os.path.exists(img_path):
            base, _ = os.path.splitext(filename)
            alt_path = os.path.join(self.img_root, base + ".png")
            if os.path.exists(alt_path):
                img_path = alt_path

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return None, None, None

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx, filename


class PairedCleanAdvDataset(Dataset):
    """
    Dataset that returns (clean_image, adv_image, label) pairs.
    
    Used for model merging optimization (e.g., genetic algorithm) and 
    analysis tasks that compare clean vs. adversarial behavior.

    Args:
        dataset_name: Name of the dataset.
        data_root: Root directory for clean data.
        adv_root: Root directory for adversarial samples.
        adv_subfolder: Model-specific subfolder (e.g., "CLIP-L-14").
        adv_eps_folder: Epsilon subfolder (e.g., "4_255").
        preprocess: Torchvision transform for image preprocessing.
        sample_n: Number of samples to use.
    """

    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        adv_root: str,
        adv_subfolder: str = "CLIP-L-14",
        adv_eps_folder: str = "4_255",
        preprocess: transforms.Compose = None,
        sample_n: int = 500,
    ):
        self.preprocess = preprocess
        self.dataset_path = os.path.join(data_root, dataset_name)

        csv_path = os.path.join(self.dataset_path, "labels.csv")
        df = pd.read_csv(csv_path)
        if len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=42)

        self.image_files = df.iloc[:, 0].values
        self.labels = df.iloc[:, 1].values

        unique_labels = sorted(list(set(self.labels)))
        self.cls_to_idx = {lb: i for i, lb in enumerate(unique_labels)}
        self.classes = unique_labels

        self.clean_root = os.path.join(self.dataset_path, "images")
        self.adv_root = os.path.join(adv_root, adv_subfolder, dataset_name, adv_eps_folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        label_text = self.labels[idx]
        label_idx = self.cls_to_idx[label_text]

        clean_path = os.path.join(self.clean_root, filename)
        adv_filename = os.path.splitext(filename)[0] + ".png"
        adv_path = os.path.join(self.adv_root, adv_filename)

        try:
            img_clean = Image.open(clean_path).convert("RGB")
            if os.path.exists(adv_path):
                img_adv = Image.open(adv_path).convert("RGB")
            else:
                img_adv = img_clean

            if self.preprocess:
                return self.preprocess(img_clean), self.preprocess(img_adv), label_idx
            return img_clean, img_adv, label_idx
        except Exception:
            return None
