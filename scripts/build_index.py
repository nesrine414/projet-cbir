"""
Phase Off-line : calcul et stockage des signatures.

Ce script parcourt toute la base data/objects/, extrait les signatures
de chaque image avec chaque descripteur, et sauvegarde l'index dans index/.

Usage :
    python scripts/build_index.py
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

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "objects"

# Tous les descripteurs disponibles
ALL_DESCRIPTORS = {
    "hsv": HSVDescriptor(),
    "rgb": RGBDescriptor(),
    "lbp": LBPDescriptor(),
    "hog": HOGDescriptor(),
}

# Extensions d'images supportées
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Fonctions ─────────────────────────────────────────────────────────────────

def collect_images(data_dir: Path) -> list[Path]:
    """Retourne la liste de toutes les images dans data/objects/."""
    images = [
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def build_index(descriptors: dict, data_dir: Path) -> dict:
    """
    Calcule les signatures de toutes les images pour chaque descripteur.

    Args:
        descriptors : {nom: instance_Descriptor}.
        data_dir    : dossier racine des images.

    Returns:
        Index complet {nom_descripteur: {chemin_relatif: vecteur}}.
    """
    images = collect_images(data_dir)

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
        "--descriptors",
        nargs="+",
        choices=list(ALL_DESCRIPTORS.keys()),
        default=list(ALL_DESCRIPTORS.keys()),
        help="Descripteurs à calculer (défaut : tous)",
    )
    parser.add_argument(
        "--output",
        default="index.pkl",
        help="Nom du fichier de sortie dans index/ (défaut : index.pkl)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Filtrer les descripteurs demandés
    selected = {k: ALL_DESCRIPTORS[k] for k in args.descriptors}

    # Construire l'index
    index = build_index(selected, DATA_DIR)

    # Résumé
    print("\n[build_index] Résumé :")
    for name, sigs in index.items():
        print(f"  {name:20s} -> {len(sigs)} signatures")

    # Sauvegarder
    save_index(index, args.output)
    print("\nIndexation terminee avec succes !")
