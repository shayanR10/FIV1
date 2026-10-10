"""CLIP-powered semantic image filtering utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_JUNK_PROMPTS = [
    "A scientific diagram, taxonomic tree, family tree chart, or text-heavy graphic",
    "An illustration, painting, digital render, or children's book page of a dinosaur",
    "A collage, museum wall layout, or photograph containing multiple different dinosaur skeletons",
    "A close-up macro shot of a single isolated bone, individual tooth, or detached claw",
]


class ClipSemanticFilter:
    """Encapsulates CLIP model setup and semantic clearing decisions."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def score_images(self, image_batch: list[Image.Image], prompts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=list(prompts), images=image_batch, return_tensors="pt", padding=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits_per_image
        return torch.softmax(logits, dim=-1)


def build_clip_filter(model_name: str = "openai/clip-vit-base-patch32", device: str | None = None) -> ClipSemanticFilter:
    """Factory helper for CLIP semantic filtering."""
    return ClipSemanticFilter(model_name=model_name, device=device)


def semantic_clear_species_images(
    species: str,
    data_dir: str | Path,
    clip_filter: ClipSemanticFilter,
    batch_size: int = 32,
) -> dict[str, int]:
    """Remove broken or semantically irrelevant images for a species directory."""
    species_dir = Path(data_dir) / species.replace(" ", "_")
    if not species_dir.exists():
        return {"broken": 0, "irrelevant": 0}

    prompts = [
        f"A clear, isolated photograph of a single complete museum dinosaur skeleton of a {species}",
        *DEFAULT_JUNK_PROMPTS,
    ]

    all_files = sorted(species_dir.glob("*.jpg"))
    broken = 0
    irrelevant = 0

    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i : i + batch_size]
        images: list[Image.Image] = []
        valid_paths: list[Path] = []

        for file_path in batch_files:
            try:
                with Image.open(file_path) as quick_check:
                    quick_check.verify()
                with Image.open(file_path) as raw_image:
                    clean_image = raw_image.convert("RGB")
                    images.append(clean_image.copy())
                valid_paths.append(file_path)
            except (OSError, ValueError, RuntimeError):
                file_path.unlink(missing_ok=True)
                broken += 1

        if not images:
            continue

        probabilities = clip_filter.score_images(images, prompts)
        top_predictions = probabilities.argmax(dim=-1).cpu().tolist()

        for index, predicted_prompt in enumerate(top_predictions):
            if predicted_prompt != 0:
                valid_paths[index].unlink(missing_ok=True)
                irrelevant += 1

    return {"broken": broken, "irrelevant": irrelevant}
