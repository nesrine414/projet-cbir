"""
Orchestrateur de la recherche CBIR (phase On-line).

Relie tous les modules :
    image requête -> descripteur -> vecteur -> index -> distances -> résultats

Fonctions exposées à l'API :
    - search()              : usage direct (chemin fichier)
    - visual_search()       : appelée par POST /search (image_id dans la base)
    - visual_search_image() : appelée par POST /search/upload (bytes uploadés)
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

    return linear.search(query_vector, desc_index, distance_fn, k=k)


def visual_search(
    dataset: str,
    query_image_id: str,
    descriptors: list[str],
    k: int = 20,
) -> list[dict]:
    """
    Recherche visuelle à partir d'une image déjà dans la base.
    Appelée par POST /search de l'API.

    Args:
        dataset         : nom du dataset ('objects').
        query_image_id  : chemin relatif de l'image dans data/ (ex: 'accordion/image_0001.jpg').
        descriptors     : liste de noms de descripteurs (ex: ['color', 'texture']).
        k               : nombre de résultats.

    Returns:
        Liste de dicts {"id": str, "distance": float}.
    """
    # Résoudre le chemin complet de l'image requête
    img_path = DATA_DIR / dataset / query_image_id
    if not img_path.exists():
        raise FileNotFoundError(f"Image introuvable : {img_path}")

    image = Image.open(img_path).convert("RGB")

    # Si plusieurs descripteurs demandés : fusionner les scores (union des résultats)
    if not descriptors:
        descriptors = ["color"]

    if len(descriptors) == 1:
        return _run_search(image, descriptors[0], k)

    # Fusion multi-descripteurs : moyenne des rangs
    from collections import defaultdict
    rank_sum: dict[str, float] = defaultdict(float)

    for desc_name in descriptors:
        results = _run_search(image, desc_name, k * 2)
        for rank, item in enumerate(results):
            rank_sum[item["id"]] += rank  # plus le rang est petit, mieux c'est

    # Trier par rang moyen croissant
    fused = sorted(rank_sum.items(), key=lambda x: x[1])[:k]
    return [{"id": img_id, "distance": float(rank)} for img_id, rank in fused]


def visual_search_image(
    dataset: str,
    image_bytes: bytes,
    descriptors: list[str],
    k: int = 20,
) -> list[dict]:
    """
    Recherche visuelle à partir d'une image uploadée (bytes bruts).
    Appelée par POST /search/upload de l'API.

    Args:
        dataset     : nom du dataset ('objects').
        image_bytes : contenu binaire de l'image uploadée.
        descriptors : liste de noms de descripteurs.
        k           : nombre de résultats.

    Returns:
        Liste de dicts {"id": str, "distance": float}.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if not descriptors:
        descriptors = ["color"]

    if len(descriptors) == 1:
        return _run_search(image, descriptors[0], k)

    # Fusion multi-descripteurs (même logique que visual_search)
    from collections import defaultdict
    rank_sum: dict[str, float] = defaultdict(float)

    for desc_name in descriptors:
        results = _run_search(image, desc_name, k * 2)
        for rank, item in enumerate(results):
            rank_sum[item["id"]] += rank

    fused = sorted(rank_sum.items(), key=lambda x: x[1])[:k]
    return [{"id": img_id, "distance": float(rank)} for img_id, rank in fused]
