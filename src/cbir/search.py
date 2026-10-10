"""
Orchestrateur de la recherche CBIR (phase On-line).

Relie tous les modules :
    image requête -> descripteur -> vecteur -> index -> distances -> résultats

Usage depuis l'API ou les scripts :
    from cbir.search import search
    results = search("data/objects/accordion/image_0001.jpg", "hsv", "chi2", k=10)
"""

import io
from pathlib import Path

from PIL import Image

from cbir.descriptors.color import HSVDescriptor, RGBDescriptor
from cbir.descriptors.texture import LBPDescriptor
from cbir.descriptors.shape import HOGDescriptor
from cbir.distances import get_distance
from cbir.index.store import load_index, get_descriptor_index
from cbir.index import linear


# ── Registres ─────────────────────────────────────────────────────────────────

# Descripteurs disponibles — nom court -> instance
DESCRIPTORS = {
    "hsv":   HSVDescriptor(),
    "rgb":   RGBDescriptor(),
    "lbp":   LBPDescriptor(),
    "hog":   HOGDescriptor(),
    # Alias utilisés par l'API / l'interface web
    "color":   HSVDescriptor(),
    "texture": LBPDescriptor(),
    "shape":   HOGDescriptor(),
}

# Distance par défaut selon le type de descripteur
DEFAULT_DISTANCE = {
    "hsv":     "chi2",
    "rgb":     "chi2",
    "color":   "chi2",
    "lbp":     "chi2",
    "texture": "chi2",
    "hog":     "l2",
    "shape":   "l2",
}

# Dossier racine du projet (pour résoudre les chemins d'images)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Cache de l'index en mémoire (évite de recharger le pickle à chaque requête)
_index_cache: dict | None = None


def _get_index(index_file: str = "index.pkl") -> dict:
    """Charge l'index une seule fois et le garde en mémoire."""
    global _index_cache
    if _index_cache is None:
        _index_cache = load_index(index_file)
    return _index_cache


def _run_search(image: Image.Image, descriptor_name: str, k: int) -> list[dict]:
    """
    Logique commune : extrait le vecteur, cherche dans l'index, retourne
    une liste de dicts {"id": chemin_relatif, "distance": float}.
    """
    if descriptor_name not in DESCRIPTORS:
        raise ValueError(
            f"Descripteur inconnu '{descriptor_name}'. "
            f"Disponibles : {list(DESCRIPTORS.keys())}"
        )

    descriptor = DESCRIPTORS[descriptor_name]
    query_vector = descriptor.extract(image)

    full_index = _get_index()
    desc_index = get_descriptor_index(full_index, descriptor.name)

    distance_fn = get_distance(DEFAULT_DISTANCE.get(descriptor_name, "chi2"))

    raw = linear.search(query_vector, desc_index, distance_fn, k=k)

    # Normaliser le séparateur pour l'ID (toujours slash, compatible URL)
    return [{"id": path.replace("\\", "/"), "distance": float(dist)}
            for path, dist in raw]


# ── API publique ───────────────────────────────────────────────────────────────

def search(
    image_path: str,
    descriptor_name: str = "hsv",
    distance_name: str = "chi2",
    k: int = 10,
    index_file: str = "index.pkl",
) -> list[tuple[str, float]]:
    """
    Recherche les k images les plus similaires à une image requête (usage direct).

    Args:
        image_path      : chemin vers l'image requête.
        descriptor_name : 'hsv', 'rgb', 'lbp' ou 'hog'.
        distance_name   : 'chi2', 'l1' ou 'l2'.
        k               : nombre de résultats.
        index_file      : nom du fichier pickle dans index/.

    Returns:
        Liste de (chemin_image, distance) triée du plus similaire au moins.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    image = Image.open(img_path).convert("RGB")
    descriptor = DESCRIPTORS[descriptor_name]
    query_vector = descriptor.extract(image)

    full_index = _get_index(index_file)
    desc_index = get_descriptor_index(full_index, descriptor.name)
    distance_fn = get_distance(distance_name)

    # 6. Recherche exhaustive et retour des résultats
    results = linear.search(query_vector, desc_index, distance_fn, k=k)

    return results
