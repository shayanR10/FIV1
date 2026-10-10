"""Dynamic fallback routing based on Paleobiology Database taxonomy."""

from __future__ import annotations

from typing import Any

import requests

PBDB_DINOSAURIA_URL = (
    "https://paleobiodb.org/data1.2/taxa/list.json?base_name=Dinosauria&status=accepted"
)


def fetch_dinosaur_taxonomy(api_url: str = PBDB_DINOSAURIA_URL, timeout: float = 30.0) -> dict[str, dict[str, Any]]:
    """Fetch and normalize taxonomy records from PBDB."""
    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()

    taxonomy_index: dict[str, dict[str, Any]] = {}
    for record in response.json().get("records", []):
        taxon_id = str(record.get("oid"))
        taxonomy_index[taxon_id] = {
            "name": record.get("nam"),
            "rank": record.get("rnk"),
            "parent": str(record.get("par")) if record.get("par") is not None else None,
        }

    return taxonomy_index


def resolve_fallback_taxon(
    taxon_id: str | int,
    taxonomy_index: dict[str, dict[str, Any]],
    preferred_ranks: tuple[str, ...] = ("family", "genus", "superfamily", "subfamily", "infraorder"),
) -> tuple[str, str]:
    """Walk up the taxonomy tree and return the first preferred rank."""
    current_id = str(taxon_id)
    record = taxonomy_index.get(current_id)
    if not record:
        return "UNKNOWN TAXON", "UNKNOWN RANK"

    parent_id = record.get("parent")
    while parent_id:
        parent = taxonomy_index.get(str(parent_id))
        if not parent:
            break
        rank = parent.get("rank")
        if rank in preferred_ranks:
            return parent.get("name") or "Unidentified Dinosauria", rank
        parent_id = parent.get("parent")

    return "Unidentified Dinosauria", "clade"


def classify_with_confidence(
    species_name: str,
    confidence_score: float,
    taxon_id: str | int,
    taxonomy_index: dict[str, dict[str, Any]],
    high_threshold: float = 0.85,
    low_threshold: float = 0.50,
) -> tuple[str | None, float, str | None]:
    """Apply confidence routing and return direct or fallback classification."""
    if confidence_score >= high_threshold:
        return species_name, confidence_score, None

    fallback_name, fallback_rank = resolve_fallback_taxon(taxon_id, taxonomy_index)
    if confidence_score >= low_threshold:
        return fallback_name, confidence_score, fallback_rank

    return None, confidence_score, fallback_rank
