from pathlib import Path

from PIL import Image

from src.filter import semantic_clear_species_images


class DummyFilter:
    def score_images(self, image_batch, prompts):
        class _Scores:
            def argmax(self, dim=-1):
                class _Values:
                    def cpu(self):
                        return self

                    def tolist(self):
                        return [0] + [1] * (len(image_batch) - 1)

                return _Values()

        return _Scores()


def test_semantic_clear_species_images_removes_irrelevant(tmp_path: Path):
    species_dir = tmp_path / "Testosaurus"
    species_dir.mkdir(parents=True, exist_ok=True)

    for index in range(2):
        image = Image.new("RGB", (10, 10), color=(index * 10, 0, 0))
        image.save(species_dir / f"{index:03d}.jpg")

    result = semantic_clear_species_images(
        species="Testosaurus",
        data_dir=tmp_path,
        clip_filter=DummyFilter(),
        batch_size=2,
    )

    assert result == {"broken": 0, "irrelevant": 1}
    assert (species_dir / "000.jpg").exists()
    assert not (species_dir / "001.jpg").exists()
