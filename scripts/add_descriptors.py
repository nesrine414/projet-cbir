"""
Ajoute des descripteurs à l'index EXISTANT sans recalculer les autres
(évite de refaire les 23 minutes de HOG).

Usage, depuis la racine du projet :
    python scripts/add_descriptors.py                 # ajoute cld, ehd, scd
    python scripts/add_descriptors.py --descriptors cld ehd
    python scripts/add_descriptors.py --output index.pkl

Si un descripteur existe déjà dans l'index, il est recalculé et remplacé.
Conseil : fais une copie de index/index.pkl avant de lancer.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from cbir.descriptors.mpeg7 import (  # noqa: E402
    ColorLayoutDescriptor, EdgeHistogramDescriptor, ScalableColorDescriptor,
)
from cbir.index.store import load_index, save_index  # noqa: E402

DATA_DIR = ROOT / "data" / "objects"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

AVAILABLE = {
    "cld": ColorLayoutDescriptor(),
    "ehd": EdgeHistogramDescriptor(),
    "scd": ScalableColorDescriptor(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptors", nargs="+", choices=list(AVAILABLE),
                    default=list(AVAILABLE))
    ap.add_argument("--output", default="index.pkl")
    args = ap.parse_args()

    index = load_index()
    selected = [AVAILABLE[k] for k in args.descriptors]
    for d in selected:
        index[d.name] = {}

    images = sorted(p for p in DATA_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"[add_descriptors] {len(images)} images, descripteurs : {[d.name for d in selected]}")

    for img_path in tqdm(images, desc="Indexation", unit="img"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[add_descriptors] Image ignorée {img_path.name} : {e}")
            continue
        # Même clé que build_index.py, pour rester alignée avec les autres descripteurs
        rel_path = str(img_path.relative_to(DATA_DIR.parent))
        for d in selected:
            try:
                index[d.name][rel_path] = d.extract(img)
            except Exception as e:
                print(f"[add_descriptors] Erreur {d.name} sur {img_path.name} : {e}")

    print("\n[add_descriptors] Résumé :")
    for name, sigs in index.items():
        print(f"  {name:22s} -> {len(sigs)} signatures")

    save_index(index, args.output)
    print("\nTerminé.")


if __name__ == "__main__":
    main()