"""FIV1 modular pipeline package."""

from .fallback import classify_with_confidence, fetch_dinosaur_taxonomy, resolve_fallback_taxon
from .filter import build_clip_filter, semantic_clear_species_images
from .scraper import DEFAULT_CORE_SPECIES, build_species_list, harvest_species_images, run_harvest_pipeline
from .train import build_transforms, train_convnext

__all__ = [
    "DEFAULT_CORE_SPECIES",
    "build_species_list",
    "harvest_species_images",
    "run_harvest_pipeline",
    "build_clip_filter",
    "semantic_clear_species_images",
    "fetch_dinosaur_taxonomy",
    "resolve_fallback_taxon",
    "classify_with_confidence",
    "build_transforms",
    "train_convnext",
]
