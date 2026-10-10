"""ConvNeXt training utilities and torchvision transforms."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Build training and validation image transforms."""
    train_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value="random"),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, val_transforms


def _build_dataloaders(
    dataset_path: str | Path,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, int]:
    train_transforms, val_transforms = build_transforms()

    train_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=dataset_path, transform=val_transforms)

    class_indices = defaultdict(list)
    for index, target in enumerate(train_dataset.targets):
        class_indices[target].append(index)

    random.seed(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for indices in class_indices.values():
        random.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices)
            continue
        split = max(1, int(len(indices) * 0.8))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    training_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(train_dataset.classes)


def _build_model(num_classes: int, device: torch.device) -> torch.nn.Module:
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[7].parameters():
        param.requires_grad = True

    features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(features, num_classes)
    return model.to(device)


def train_convnext(
    dataset_path: str | Path,
    epochs: int = 25,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    seed: int = 42,
    best_model_path: str | Path = "convnexttinyDINOSAUR_best.safetensors",
    final_model_path: str | Path = "convnexttinyDINOSAUR.safetensors",
) -> dict[str, float]:
    """Train ConvNeXt-Tiny on an ImageFolder dataset and persist model weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_classes = _build_dataloaders(dataset_path, batch_size, seed)

    model = _build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_accuracy = 0.0

    for _ in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, dim=1)
                val_correct += torch.sum(predictions == labels).item()
                val_total += labels.size(0)

        val_accuracy = (val_correct / val_total) * 100 if val_total else 0.0
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_file(model.state_dict(), str(best_model_path))

    save_file(model.state_dict(), str(final_model_path))

    return {
        "best_val_accuracy": best_val_accuracy,
        "num_classes": float(num_classes),
        "epochs": float(epochs),
    }
