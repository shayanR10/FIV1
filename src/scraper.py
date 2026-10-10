"""Wikimedia ingestion and data harvesting helpers."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

import requests

DEFAULT_CORE_SPECIES = [
    "Stegosaurus", "Kentrosaurus", "Hesperosaurus", "Tuojiangosaurus",
    "Miragaia", "Dacentrurus", "Edmontosaurus", "Saurolophus", "Maiasaura",
    "Brachylophosaurus", "Gryposaurus", "Prosaurolophus", "Shantungosaurus",
    "Parasaurolophus", "Corythosaurus", "Lambeosaurus", "Hypacrosaurus",
    "Olorotitan", "Tsintaosaurus", "Dryosaurus", "Ouranosaurus", "Iguanodon",
    "Mantellisaurus", "Protohadros", "Probactrosaurus", "Bactrosaurus",
    "Eolambia", "Tethyshadros", "Telmatosaurus", "Diplodocus", "Apatosaurus",
    "Brontosaurus", "Dicraeosaurus", "Amargasaurus", "Camarasaurus",
    "Giraffatitan", "Brachiosaurus", "Patagotitan", "Dreadnoughtus",
    "Rapetosaurus", "Saltasaurus", "Futalognkosaurus", "Shunosaurus",
    "Mamenchisaurus", "Ankylosaurus", "Zuul", "Euoplocephalus", "Pinacosaurus",
    "Saichania", "Tarchia", "Anodontosaurus", "Scolosaurus", "Akainacephalus",
    "Gobisaurus", "Shamosaurus", "Jinyunpelta", "Baryonyx", "Suchomimus",
    "Spinosaurus", "Ceratosuchops", "Riparovenator", "Irritator", "Spinomimus",
    "Dromaeosaurus", "Utahraptor", "Deinonychus", "Achillobator", "Velociraptor",
    "Tsaagan", "Linheraptor", "Microraptor", "Graciliraptor", "Sinornithosaurus",
    "Changyuraptor", "Buitreraptor", "Halszkaraptor", "Sinornithoides",
    "Gobivenator", "Mei", "Coelophysis", "Procompsognathus", "Dilophosaurus",
    "Ceratosaurus", "Carnotaurus", "Majungasaurus", "Megaraptor", "Allosaurus",
    "Sinraptor", "Monolophosaurus", "Acrocanthosaurus", "Giganotosaurus",
    "Carcharodontosaurus", "Tyrannosaurus", "Tarbosaurus", "Albertosaurus",
    "Gorgosaurus", "Daspletosaurus", "Yutyrannus", "Guanlong", "Compsognathus",
    "Sinosauropteryx", "Sinocalliopteryx", "Huaxiagnathus", "Gallimimus",
    "Struthiomimus", "Ornithomimus", "Therizinosaurus", "Alxasaurus",
    "Erlikosaurus", "Anzu", "Citipati", "Khaan", "Conchoraptor", "Caudipteryx",
    "Triceratops", "Torosaurus", "Chasmosaurus", "Anchiceratops", "Arrhinoceratops",
    "Kosmoceratops", "Utahceratops", "Pentaceratops", "Centrosaurus",
    "Styracosaurus", "Pachyrhinosaurus", "Nasutoceratops", "Einiosaurus",
    "Diabloceratops", "Wendiceratops", "Medusaceratops", "Sinoceratops",
    "Protoceratops", "Psittacosaurus",
]

EXCLUDE_WORDS = (
    "acta", "journal", "annals", "bulletin", "list", "formation", "museum",
    "paleont", "suidae", "family", "cf.", "indet", "<span>", "http", "doi",
)


def build_species_list(
    limit: int,
    kaggle_username: str,
    baseline_species: Iterable[str] = DEFAULT_CORE_SPECIES,
    request_timeout: float = 30.0,
) -> list[str]:
    """Build a hybrid species list from curated taxa and Wikimedia categories."""
    species = list(dict.fromkeys(baseline_species))
    if len(species) >= limit:
        return species[:limit]

    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Dinosaur_genera",
            "cmlimit": 500,
            "cmtype": "page",
            "format": "json",
        },
        headers={
            "User-Agent": f"dinodatafinder/1.0 (https://kaggle.com/{kaggle_username}) Python-requests",
        },
        timeout=request_timeout,
    )
    response.raise_for_status()

    for page in response.json().get("query", {}).get("categorymembers", []):
        title = page.get("title", "").strip()
        if not title:
            continue
        genus_name = title.split()[0].capitalize()
        if not genus_name or not genus_name[0].isalpha():
            continue
        lower_title = title.lower()
        if any(token in lower_title for token in EXCLUDE_WORDS):
            continue
        if lower_title.endswith(("idae", "inae", "osauria")):
            continue
        if genus_name not in species:
            species.append(genus_name)
            if len(species) >= limit:
                break

    return species[:limit]


def _download_single_image(
    image_url: str,
    image_path: Path,
    session: requests.Session,
    max_size_mb: int = 15,
    throttle_seconds: float = 0.1,
) -> bool:
    if image_path.exists():
        return True

    time.sleep(throttle_seconds)
    max_bytes = max_size_mb * 1024 * 1024

    try:
        with session.get(image_url, timeout=(5, 10), stream=True) as response:
            if response.status_code != 200:
                return False

            header_size = int(response.headers.get("Content-Length", 0))
            if header_size > max_bytes:
                return False

            downloaded_bytes = 0
            with image_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > max_bytes:
                        handle.close()
                        image_path.unlink(missing_ok=True)
                        return False
                    handle.write(chunk)
        return True
    except requests.RequestException:
        image_path.unlink(missing_ok=True)
        return False


def harvest_species_images(
    species: str,
    output_dir: str | Path,
    kaggle_username: str,
    max_images: int = 200,
    max_offset: int = 300,
    max_workers: int = 10,
) -> dict[str, int]:
    """Harvest Wikimedia Commons images for one species."""
    species_folder = Path(output_dir) / species.replace(" ", "_")
    species_folder.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(species_folder.glob("*.jpg"))
    if len(existing_files) >= max_images:
        return {"saved": len(existing_files), "skipped": 0}

    query_list = [
        f"{species} skeleton",
        f"{species} fossil",
        f"{species} specimen",
        f"{species} skull",
        f"{species} holotype",
        f"{species} bones",
        f"{species} mount",
        f"{species} museum",
        species,
    ]

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"dinodatafinder/1.0 (https://kaggle.com/{kaggle_username}) Python-requests",
            "Referer": "https://commons.wikimedia.org/",
        }
    )

    saved = len(existing_files)
    skipped = 0
    download_tasks: list[tuple[str, Path]] = []

    for query in query_list:
        if saved + len(download_tasks) >= max_images:
            break

        offset = 0
        while saved + len(download_tasks) < max_images and offset < max_offset:
            response = session.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": query,
                    "gsrnamespace": "6",
                    "gsrlimit": 50,
                    "gsroffset": offset,
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata",
                    "format": "json",
                },
                timeout=10,
            )
            payload = response.json()
            pages = payload.get("query", {}).get("pages", {})
            if not pages:
                break

            for _, image_info in pages.items():
                if saved + len(download_tasks) >= max_images:
                    break
                if "imageinfo" not in image_info:
                    continue

                info = image_info["imageinfo"][0]
                image_url = info.get("url", "")
                path = urlparse(image_url).path.lower()
                if not path.endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue

                metadata = info.get("extmetadata", {})
                metadata_text = (
                    str(metadata.get("ImageDescription", {}).get("value", ""))
                    + str(metadata.get("Categories", {}).get("value", ""))
                ).lower()
                if any(token in metadata_text for token in ("toy", "figure", "lego", "cartoon", "logo")):
                    skipped += 1
                    continue

                target_path = species_folder / f"{saved + len(download_tasks):03d}.jpg"
                download_tasks.append((image_url, target_path))

            offset += 50

    if download_tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_download_single_image, image_url, image_path, session)
                for image_url, image_path in download_tasks
            ]
            for future in futures:
                if future.result():
                    saved += 1

    return {"saved": saved, "skipped": skipped}


def run_harvest_pipeline(
    species_list: Iterable[str],
    output_dir: str | Path,
    kaggle_username: str,
    max_images_per_species: int = 200,
    post_process: Callable[[str, str | Path], dict[str, int]] | None = None,
) -> dict[str, dict[str, int]]:
    """Run ingestion for a species list and optionally apply post-processing per species."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {}
    for species in species_list:
        stats = harvest_species_images(
            species=species,
            output_dir=output_path,
            kaggle_username=kaggle_username,
            max_images=max_images_per_species,
        )
        if post_process:
            filter_stats = post_process(species, output_path)
            stats.update({f"filter_{k}": v for k, v in filter_stats.items()})
        summary[species] = stats

    return summary
