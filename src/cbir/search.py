"""
Orchestrateur de la recherche CBIR (phase On-line).

Relie tous les modules :
    image requête -> descripteur -> vecteur -> index -> distances -> résultats

Usage depuis l'API ou les scripts :
    from cbir.search import search, visual_search
    results = search("data/objects/accordion/image_0001.jpg", "hsv", "chi2", k=10)
"""

from pathlib import Path
from PIL import Image

from cbir.descriptors.color import HSVDescriptor, RGBDescriptor
from cbir.descriptors.texture import LBPDescriptor
from cbir.descriptors.shape import HOGDescriptor
from cbir.distances import get_distance
from cbir.index.store import load_index, get_descriptor_index
from cbir.index import linear


# Registre des descripteurs disponibles
DESCRIPTORS = {
    "hsv": HSVDescriptor(),
    "rgb": RGBDescriptor(),
    "lbp": LBPDescriptor(),
    "hog": HOGDescriptor(),
}

# Cache de l'index en mémoire (évite de recharger le pickle à chaque requête)
_index_cache: dict | None = None


def _get_index(index_file: str = "index.pkl") -> dict:
    """Charge l'index une seule fois et le garde en mémoire."""
    global _index_cache
    if _index_cache is None:
        _index_cache = load_index(index_file)
    return _index_cache


def search(
    image_path: str,
    descriptor_name: str = "hsv",
    distance_name: str = "chi2",
    k: int = 10,
    index_file: str = "index.pkl",
) -> list[tuple[str, float]]:
    """
    Recherche les k images les plus similaires à une image requête.

    Args:
        image_path      : chemin vers l'image requête.
        descriptor_name : 'hsv', 'rgb', 'lbp' ou 'hog'.
        distance_name   : 'chi2', 'l1' ou 'l2'.
        k               : nombre de résultats.
        index_file      : nom du fichier pickle dans index/.

    Returns:
        Liste de (chemin_image, distance) triée du plus similaire au moins.

    Raises:
        FileNotFoundError : si l'image ou l'index n'existe pas.
        KeyError          : si le descripteur n'est pas dans l'index.
    """
    # 1. Vérifier que le descripteur existe
    if descriptor_name not in DESCRIPTORS:
        raise ValueError(
            f"Descripteur inconnu '{descriptor_name}'. "
            f"Disponibles : {list(DESCRIPTORS.keys())}"
        )

    # 2. Charger l'image requête
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    image = Image.open(img_path).convert("RGB")

    # 3. Extraire la signature de la requête
    descriptor = DESCRIPTORS[descriptor_name]
    query_vector = descriptor.extract(image)

    # 4. Charger l'index et récupérer la partie du bon descripteur
    full_index = _get_index(index_file)
    desc_index = get_descriptor_index(full_index, descriptor.name)

    # 5. Récupérer la fonction de distance
    distance_fn = get_distance(distance_name)

    # 6. Recherche exhaustive et retour des résultats
    results = linear.search(query_vector, desc_index, distance_fn, k=k)

    return results


def visual_search(dataset: str, query_id: str, descriptors: list[str] | str, k: int = 20, metric: str = "chi2"):
    """
    Retourne [{"id": str, "distance": float}], du plus proche au plus éloigné.
    Contrat avec l'API (api/main.py).
    """
    desc_map = {"color": "hsv", "texture": "lbp", "shape": "hog"}
    if isinstance(descriptors, list) and descriptors:
        desc_name = descriptors[0]
    elif isinstance(descriptors, str):
        desc_name = descriptors
    else:
        desc_name = "hsv"
    desc_name = desc_map.get(desc_name, desc_name)

    dist_map = {"l1": "l1", "l2": "l2", "chi2": "chi2"}
    dist_name = dist_map.get(metric, "chi2")

    img_path = Path("data") / dataset / query_id
    try:
        raw_results = search(str(img_path), descriptor_name=desc_name, distance_name=dist_name, k=k)
        results = []
        for p, d in raw_results:
            clean_id = p.replace(f"{dataset}/", "").replace(f"{dataset}\\", "")
            results.append({"id": clean_id, "distance": float(d)})
        return results
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise NotImplementedError(f"Recherche visuelle non disponible : {e}")
