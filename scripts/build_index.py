"""
Phase Off-line : calcul et stockage des signatures.

Ce script parcourt toute la base data/objects/, extrait les signatures
de chaque image avec chaque descripteur (classiques + deep ResNet),
et sauvegarde l'index dans index/.

Usage :
    python scripts/build_index.py
"""

import argparse
import sys
from pathlib import Path

# Ajouter src/ au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image
from tqdm import tqdm

from cbir.descriptors.color import HSVDescriptor
from cbir.descriptors.mpeg7 import (
    ColorLayoutDescriptor,
    EdgeHistogramDescriptor,
    ScalableColorDescriptor,
)
from cbir.descriptors.texture import LBPDescriptor
from cbir.descriptors.shape import HOGDescriptor
from cbir.index.store import load_index, save_index

# ── Configuration ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
DATASET_ROOTS = {
    "objects": DATA_ROOT / "objects",
    "mirflickr": DATA_ROOT / "mirflickr25k",
}

BASE_DESCRIPTORS = {
    "hsv": HSVDescriptor(),
    "lbp": LBPDescriptor(),
    "hog": HOGDescriptor(),
    "cld": ColorLayoutDescriptor(),
    "ehd": EdgeHistogramDescriptor(),
    "scd": ScalableColorDescriptor(),
}
ALL_DESCRIPTOR_NAMES = [*BASE_DESCRIPTORS, "resnet"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(data_dir: Path) -> list[Path]:
    """Retourne la liste de toutes les images dans data/objects/."""
    images = [
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def build_index(descriptors: dict, data_dir: Path, dataset: str | None = None) -> dict:
    """
    Calcule les signatures de toutes les images pour chaque descripteur.
    Utilise le batching pour accélérer l'extraction des réseaux profonds (ResNet).
    """
    images = collect_images(data_dir)
    dataset_prefix = dataset or data_dir.name

    if not images:
        print(f"[build_index] Aucune image trouvée dans {data_dir}")
        sys.exit(1)

    print(f"[build_index] {len(images)} images trouvées dans {data_dir}")
    print(f"[build_index] Descripteurs : {list(descriptors.keys())}\n")

    index = {desc.name: {} for desc in descriptors.values()}

    batch_size = 32
    for b_start in tqdm(range(0, len(images), batch_size), desc="Indexation par batch"):
        batch_paths = images[b_start : b_start + batch_size]
        batch_imgs = []
        valid_rel_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
                valid_rel_paths.append(f"{dataset_prefix}/{p.relative_to(data_dir).as_posix()}")
            except Exception as e:
                print(f"[build_index] Image ignorée {p.name} : {e}")

        if not batch_imgs:
            continue

        for desc in descriptors.values():
            if hasattr(desc, "extract_batch"):
                embs = desc.extract_batch(batch_imgs)
                for rp, emb in zip(valid_rel_paths, embs):
                    index[desc.name][rp] = emb
            else:
                for rp, img in zip(valid_rel_paths, batch_imgs):
                    try:
                        index[desc.name][rp] = desc.extract(img)
                    except Exception as e:
                        print(f"[build_index] Erreur {desc.name} sur {rp} : {e}")

    return index


def parse_args():
    parser = argparse.ArgumentParser(description="Build a visual descriptor index for one image dataset")
    parser.add_argument(
        "--dataset",
        choices=("objects", "mirflickr"),
        default="objects",
        help="Dataset folder under data/ (default: objects)",
    )
    parser.add_argument(
        "--descriptors",
        nargs="+",
        choices=ALL_DESCRIPTOR_NAMES,
        default=list(BASE_DESCRIPTORS),
        help="Descriptors to calculate (default: HSV, LBP, HOG and MPEG-7)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename under index/ (default: index.pkl for Objects, mirflickr.pkl for MIRFLICKR)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing signatures and replace only the selected descriptors",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = {name: BASE_DESCRIPTORS[name] for name in args.descriptors if name != "resnet"}
    if "resnet" in args.descriptors:
        try:
            from cbir.descriptors.deep import ResNetDescriptor
        except ImportError as exc:
            raise SystemExit("Install the optional torch and torchvision packages to index ResNet features.") from exc
        selected["resnet"] = ResNetDescriptor()

    dataset_dir = DATASET_ROOTS[args.dataset]
    output = args.output or ("index.pkl" if args.dataset == "objects" else f"{args.dataset}.pkl")
    index = build_index(selected, dataset_dir, args.dataset)

    if args.append:
        try:
            existing = load_index(output)
        except FileNotFoundError:
            existing = {}
        existing.update(index)
        index = existing

    print("\n[build_index] Résumé :")
    for name, signatures in index.items():
        print(f"  {name:<20} -> {len(signatures)} signatures")

    out_path = save_index(index, filename=output)
    print(f"[store] Index sauvegardé : {out_path}\n")
    print("Indexation terminée avec succès !")
