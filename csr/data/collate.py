"""
Custom collate functions for DataLoader.
"""

import torch


def default_collate_fn(batch):
    """
    Collate for CLIPDataset: filters out None items.
    Returns (images, labels, filenames) — images are PIL or Tensor depending on transform.
    """
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None, None

    images = [item[0] for item in batch]
    # If images are already tensors (transform applied), stack them
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
        labels = torch.tensor([item[1] for item in batch])
    else:
        labels = torch.tensor([item[1] for item in batch])

    filenames = [item[2] for item in batch]
    return images, labels, filenames


def paired_collate_fn(batch):
    """
    Collate for PairedCleanAdvDataset: filters None and stacks tensors.
    Returns (clean_imgs, adv_imgs, labels).
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None
    cleans = torch.stack([b[0] for b in batch])
    advs = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch])
    return cleans, advs, labels
