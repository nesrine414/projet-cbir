"""
Phase Off-line : calcul et stockage des signatures.

Ce script parcourt le dataset sélectionné, extrait les signatures de chaque
image avec chaque descripteur, et sauvegarde l'index dans index/.

Usage :
    python scripts/build_index.py
    python scripts/build_index.py --dataset mirflickr
    python scripts/build_index.py --descriptors hsv lbp
    python scripts/build_index.py --output my_index.pkl
"""

import sys
import argparse
from pathlib import Path

# Ajouter src/ au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image
from tqdm import tqdm

from cbir.descriptors.color import HSVDescriptor, RGBDescriptor
from cbir.descriptors.texture import LBPDescriptor
from cbir.descriptors.shape import HOGDescriptor
from cbir.index.store import save_index
from cbir.descriptors.spatial import HSVGridDescriptor, LBPGridDescriptor

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRS = {
    "objects": PROJECT_ROOT / "data" / "objects",
    "mirflickr": PROJECT_ROOT / "data" / "mirflickr",
}
INDEX_FILES = {
    "objects": "index.pkl",
    "mirflickr": "mirflickr.pkl",
}

# Tous les descripteurs disponibles
ALL_DESCRIPTORS = {
    "hsv": HSVDescriptor(),
    "rgb": RGBDescriptor(),
    "lbp": LBPDescriptor(),
    "hog": HOGDescriptor(),
    "hsvgrid": HSVGridDescriptor(),
    "lbpgrid": LBPGridDescriptor(),
}

# Extensions d'images supportées
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Fonctions ─────────────────────────────────────────────────────────────────

def collect_images(data_dir: Path) -> list[Path]:
    """Retourne la liste de toutes les images dans le dataset."""
    images = [
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def select_images(images: list[Path], data_dir: Path, limit: int) -> list[Path]:
    """Sélectionne de manière déterministe un échantillon réparti entre les shards."""
    groups: dict[str, list[Path]] = {}
    for image_path in images:
        parts = image_path.relative_to(data_dir).parts
        group = parts[1] if len(parts) > 2 and parts[0] == "thumbnails" else parts[0]
        groups.setdefault(group, []).append(image_path)

    selected = []
    group_names = sorted(groups)
    base_count, remainder = divmod(limit, len(group_names))
    for group_index, group_name in enumerate(group_names):
        group_images = groups[group_name]
        count = min(base_count + (group_index < remainder), len(group_images))
        if count == 0:
            continue
        step = len(group_images) / count
        selected.extend(
            group_images[min(int((sample_index + 0.5) * step), len(group_images) - 1)]
            for sample_index in range(count)
        )
    return sorted(selected)


def build_index(descriptors: dict, data_dir: Path, limit: int | None = None) -> dict:
    """
    Calcule les signatures de toutes les images pour chaque descripteur.

    Args:
        descriptors : {nom: instance_Descriptor}.
        data_dir    : dossier racine des images.

    Returns:
        Index complet {nom_descripteur: {chemin_relatif: vecteur}}.
    """
    images = collect_images(data_dir)
    total_images = len(images)
    if limit is not None and total_images > limit:
        images = select_images(images, data_dir, limit)
        print(
            f"[build_index] Échantillon équilibré : {len(images)} images "
            f"sur {total_images}"
        )

    if not images:
        print(f"[build_index] Aucune image trouvée dans {data_dir}")
        sys.exit(1)

    print(f"[build_index] {len(images)} images trouvées dans {data_dir}")
    print(f"[build_index] Descripteurs : {list(descriptors.keys())}\n")

    # Initialiser l'index vide — clé = desc.name (ex: 'hsv_8_4_4')
    index = {desc.name: {} for desc in descriptors.values()}

    for img_path in tqdm(images, desc="Indexation", unit="img"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[build_index] Image ignoree {img_path.name} : {e}")
            continue

        # Clé relative pour la portabilité (indépendant du chemin absolu)
        rel_path = str(img_path.relative_to(data_dir.parent))

        for desc in descriptors.values():
            try:
                vector = desc.extract(img)
                index[desc.name][rel_path] = vector
            except Exception as e:
                print(f"[build_index] Erreur {desc.name} sur {img_path.name} : {e}")

    return index


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Construction de l'index CBIR (phase off-line)")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_DIRS),
        default="objects",
        help="Base à indexer (défaut : objects)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Dossier des images (remplace le chemin par défaut du dataset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Nombre maximal d'images à indexer, échantillonnées entre les shards",
    )
    parser.add_argument(
        "--descriptors",
        nargs="+",
        choices=list(ALL_DESCRIPTORS.keys()),
        default=list(ALL_DESCRIPTORS.keys()),
        help="Descripteurs à calculer (défaut : tous)",
    )
    parser.add_argument(
        "--output",
        help="Nom du fichier de sortie dans index/ (défaut : spécifique au dataset)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit doit être supérieur à 0")

    # Filtrer les descripteurs demandés
    selected = {k: ALL_DESCRIPTORS[k] for k in args.descriptors}

    # Construire l'index
    data_dir = args.root or DATASET_DIRS[args.dataset]
    index = build_index(selected, data_dir, args.limit)

    # Résumé
    print("\n[build_index] Résumé :")
    for name, sigs in index.items():
        print(f"  {name:20s} -> {len(sigs)} signatures")

    # Sauvegarder
    save_index(index, args.output or INDEX_FILES[args.dataset])
    print("\nIndexation terminee avec succes !")
