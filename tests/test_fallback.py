from src.fallback import classify_with_confidence, resolve_fallback_taxon


def test_resolve_fallback_taxon_prefers_genus_or_family():
    taxonomy = {
        "10": {"name": "SpeciesX", "rank": "species", "parent": "9"},
        "9": {"name": "GenusX", "rank": "genus", "parent": "8"},
        "8": {"name": "FamilyX", "rank": "family", "parent": None},
    }
    assert resolve_fallback_taxon("10", taxonomy) == ("GenusX", "genus")


def test_classify_with_confidence_routes_correctly():
    taxonomy = {
        "10": {"name": "SpeciesX", "rank": "species", "parent": "8"},
        "8": {"name": "FamilyX", "rank": "family", "parent": None},
    }
    assert classify_with_confidence("SpeciesX", 0.90, "10", taxonomy) == ("SpeciesX", 0.90, None)
    assert classify_with_confidence("SpeciesX", 0.60, "10", taxonomy) == ("FamilyX", 0.60, "family")
    assert classify_with_confidence("SpeciesX", 0.20, "10", taxonomy) == (None, 0.20, "family")
