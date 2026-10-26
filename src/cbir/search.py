"""
Orchestrateur CBIR — Version v6 (descripteurs spatiaux + MPEG-7).

Fusion par score combiné normalisé :
  1. Chaque descripteur calcule ses distances (chi2 pour les histogrammes,
     L2 pour HOG et CLD, L1 pour EHD et SCD)
  2. Chaque distance est divisée par la distance médiane typique du
     descripteur (NORMALIZATION = "scale", indépendante de la requête)
  3. Score final = moyenne pondérée des distances normalisées (0 = meilleur)
  4. Poids choisis avec scripts/evaluate.py (HOG et EHD dominants)

Historique :
  - v5.1 : verrou sur le cache, anti path-traversal, filtrage candidats rapide
  - v6   : normalisation par échelle, descripteurs grille, descripteurs MPEG-7
"""

import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from cbir.descriptors.color import HSVDescriptor, RGBDescriptor
from cbir.descriptors.texture import LBPDescriptor
from cbir.descriptors.shape import HOGDescriptor
from cbir.descriptors.spatial import HSVGridDescriptor, LBPGridDescriptor
from cbir.descriptors.mpeg7 import (
    ColorLayoutDescriptor, EdgeHistogramDescriptor, ScalableColorDescriptor,
)
from cbir.index.store import load_index


# ── Configuration ─────────────────────────────────────────────────────────────

# (index_key, descriptor_instance, distance_type, weight)
# Les descripteurs absents de l'index sont ignorés automatiquement (les grilles
# après build_index.py, MPEG-7 après add_descriptors.py).
# Poids : issus de scripts/evaluate.py (précision@10 sur 9 descripteurs).
# Meilleure combinaison trouvée : HOG x2 + EHD x2 + SCD x0.5 = 0.438.
# Les autres descripteurs gardent un petit poids POSITIF (0.25) : un poids nul
# sur tous les descripteurs cochés ferait une division par zéro, et chaque case
# de l'interface doit rester utilisable seule.
DESCRIPTOR_CONFIG = [
    ("hsv_8_4_4",            HSVDescriptor(),       "chi2", 0.25),
    ("rgb_8_8_8",            RGBDescriptor(),       "chi2", 0.25),
    ("lbp_P24_R3",           LBPDescriptor(),       "chi2", 0.25),
    ("hog_o9_c8x8_b2x2",     HOGDescriptor(),       "l2",   2.0),
    ("hsv_grid_3x3_8_3_3",   HSVGridDescriptor(),   "chi2", 0.25),
    ("lbp_grid_3x3_P24_R3",  LBPGridDescriptor(),   "chi2", 0.25),
    # MPEG-7 (CLD : poids de la norme déjà dans le vecteur -> L2)
    ("cld_6_3_3",            ColorLayoutDescriptor(),       "l2",   0.25),
    ("ehd_4x4_e5",           EdgeHistogramDescriptor(),     "l1",   2.0),
    ("scd_16_4_4",           ScalableColorDescriptor(),     "l1",   0.5),
]

# "scale"  : distance / distance médiane typique du descripteur (calculée une
#            fois sur l'index) -> indépendant de la requête et des outliers.
# "minmax" : ancienne méthode, relative à chaque requête.
NORMALIZATION = "scale"

# Re-ranking par vote de catégorie (recherche par image seule uniquement).
# Les CATEGORY_VOTE_K images visuellement les plus proches "votent" pour leur
# catégorie (1er dossier de l'identifiant). Les images des catégories les plus
# votées voient leur score réduit de jusqu'à CATEGORY_BETA * 100 %.
# Réglage : scripts/evaluate.py (section "Re-ranking par catégorie").
# Les étiquettes viennent des noms de dossiers de la base, pas de l'image requête.
CATEGORY_RERANK = True
# Évalué sur 2 échantillons indépendants de 1000 requêtes (seeds 1 et 2), K=7, beta=0.95 :
# P@10 = 0.537 et 0.519, contre 0.449 et 0.447 sans vote.
CATEGORY_VOTE_K = 7
CATEGORY_BETA = 0.95

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_INDEX_FILES = {
    "objects": "index.pkl",
    "mirflickr": "mirflickr.pkl",
}


# ── Identifiants ──────────────────────────────────────────────────────────────

def _clean_id(raw_path: str, dataset: str | None = None) -> str:
    p = raw_path.replace("\\", "/")
    prefixes = (f"{dataset}/",) if dataset else ("objects/", "mirflickr/")
    for prefix in prefixes:
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return p


# ── Cache ─────────────────────────────────────────────────────────────────────

class _IndexCache:
    def __init__(self, dataset: str, index_filename: str):
        self.dataset = dataset
        self.index_filename = index_filename
        self.loaded = False
        self.descriptors = []   # liste de dicts
        self.all_paths = []     # identifiants ordonnés
        self.path_to_pos = {}   # identifiant -> position dans all_paths
        self.n = 0
        self.cat_codes = np.zeros(0, dtype=np.int32)  # catégorie de chaque image
        self.n_cats = 0
        self._lock = threading.Lock()

    def ensure_loaded(self):
        if self.loaded:
            return
        with self._lock:
            if self.loaded:          # double-check après acquisition du verrou
                return
            self._load()
            self.loaded = True

    def _load(self):
        raw_index = load_index(self.index_filename)

        available = [
            (key, desc, dist, w)
            for key, desc, dist, w in DESCRIPTOR_CONFIG
            if key in raw_index and len(raw_index[key]) > 0
        ]
        if not available:
            return

        # Images communes à tous les descripteurs disponibles
        common = None
        for key, _, _, _ in available:
            ids = {_clean_id(p, self.dataset) for p in raw_index[key].keys()}
            common = ids if common is None else common & ids

        all_paths = sorted(common)
        path_to_pos = {p: i for i, p in enumerate(all_paths)}

        # Construire les matrices alignées (dans une liste locale, publiée à la fin)
        infos = []
        for key, descriptor, dist_type, weight in available:
            clean_map = {
                _clean_id(p, self.dataset): v for p, v in raw_index[key].items()
            }
            matrix = np.array([clean_map[p] for p in all_paths], dtype=np.float32)

            info = {
                "key": key,
                "matrix": matrix,
                "dist_type": dist_type,
                "descriptor": descriptor,
                "weight": weight,
            }
            if dist_type == "l2":
                info["norms_sq"] = np.sum(matrix ** 2, axis=1)
            info["scale"] = _typical_distance(info)
            infos.append(info)

        _, cat_codes = np.unique([p.split("/")[0] for p in all_paths], return_inverse=True)

        self.all_paths = all_paths
        self.path_to_pos = path_to_pos
        self.n = len(all_paths)
        self.cat_codes = cat_codes.astype(np.int32)
        self.n_cats = int(cat_codes.max()) + 1 if cat_codes.size else 0
        self.descriptors = infos


_cache = _IndexCache("objects", DATASET_INDEX_FILES["objects"])
_dataset_caches = {"objects": _cache}
_dataset_caches_lock = threading.Lock()


def _cache_for_dataset(dataset: str) -> _IndexCache:
    if dataset not in DATASET_INDEX_FILES:
        raise ValueError(f"Dataset inconnu : {dataset}")
    with _dataset_caches_lock:
        if dataset not in _dataset_caches:
            _dataset_caches[dataset] = _IndexCache(
                dataset, DATASET_INDEX_FILES[dataset]
            )
        return _dataset_caches[dataset]


# ── Distances vectorisées ─────────────────────────────────────────────────────

def _compute_distances(query: np.ndarray, info: dict) -> np.ndarray:
    q = query.astype(np.float32)
    matrix = info["matrix"]
    dt = info["dist_type"]

    if dt == "chi2":
        diff = matrix - q
        summ = matrix + q + 1e-10
        return np.sum(diff ** 2 / summ, axis=1)
    elif dt == "l1":
        return np.sum(np.abs(matrix - q), axis=1)
    elif dt == "l2":  # via BLAS
        q_nsq = np.dot(q, q)
        dots = matrix @ q
        dsq = info["norms_sq"] + q_nsq - 2.0 * dots
        np.maximum(dsq, 0.0, out=dsq)
        return np.sqrt(dsq)
    raise ValueError(f"Type de distance inconnu : {dt}")


def _typical_distance(info: dict, n_samples: int = 40) -> float:
    """
    Distance médiane entre images de la base (échantillon aléatoire fixe).
    Sert d'échelle pour rendre les descripteurs comparables sans dépendre
    de la requête (contrairement au min-max).
    """
    matrix = info["matrix"]
    rng = np.random.default_rng(0)
    picks = rng.choice(len(matrix), size=min(n_samples, len(matrix)), replace=False)
    d = np.concatenate([_compute_distances(matrix[i], info) for i in picks])
    d = d[d > 1e-10]  # ignorer les distances nulles (image comparée à elle-même)
    return float(np.median(d)) if d.size else 1.0


def _category_rerank(
    fused: np.ndarray, idx: np.ndarray, cache: _IndexCache
) -> np.ndarray:
    """
    Vote de catégorie : réduit le score des images dont la catégorie domine
    parmi les CATEGORY_VOTE_K plus proches voisines. Reste positif (0 reste 0).
    """
    codes = cache.cat_codes[idx]
    # On prend K+1 voisines puis on écarte l'image requête si elle est dans la
    # base (score ~ 0) : ainsi K voisines votent, comme dans evaluate.py.
    k1 = min(CATEGORY_VOTE_K + 1, fused.size)
    top = np.argpartition(fused, k1 - 1)[:k1]
    top = top[np.argsort(fused[top])]
    top = top[fused[top] > 0.01][:CATEGORY_VOTE_K]
    if top.size == 0:
        return fused
    shares = np.bincount(codes[top], minlength=cache.n_cats) / top.size
    return fused * (1.0 - CATEGORY_BETA * shares[codes])


# ── Recherche principale ──────────────────────────────────────────────────────

_DESCRIPTOR_KEYS = {
    "color": {"hsv_8_4_4", "rgb_8_8_8", "hsv_grid_3x3_8_3_3"},
    "hsv": {"hsv_8_4_4"},
    "rgb": {"rgb_8_8_8"},
    "texture": {"lbp_P24_R3", "lbp_grid_3x3_P24_R3"},
    "lbp": {"lbp_P24_R3"},
    "hsvgrid": {"hsv_grid_3x3_8_3_3"},
    "lbpgrid": {"lbp_grid_3x3_P24_R3"},
    "mpeg7": {"cld_6_3_3", "ehd_4x4_e5", "scd_16_4_4"},
    "cld": {"cld_6_3_3"},
    "ehd": {"ehd_4x4_e5"},
    "scd": {"scd_16_4_4"},
    "shape": {"hog_o9_c8x8_b2x2"},
    "hog": {"hog_o9_c8x8_b2x2"},
}


def _optimized_search(
    image: Image.Image,
    k: int,
    descriptors: list[str] | None = None,
    candidate_ids: list[str] | set[str] | None = None,
    dataset: str = "objects",
) -> list[dict]:
    """
    Recherche par fusion pondérée multi-descripteurs.
    Avec NORMALIZATION = "scale", `distance` ~ 1 correspond à une distance
    typique de la base et 0 à une image identique. Elle reste surtout utile
    pour classer les résultats d'une même requête.
    """
    cache = _cache_for_dataset(dataset)
    cache.ensure_loaded()
    if not cache.descriptors:
        raise RuntimeError(
            f"Index vide pour {dataset}. "
            f"Lance : python scripts/build_index.py --dataset {dataset}"
        )

    selected_keys = None
    if descriptors and "auto" not in descriptors:
        selected_keys = set()
        for name in descriptors:
            keys = _DESCRIPTOR_KEYS.get(name.lower())
            if keys is None:
                raise ValueError(f"Descripteur inconnu : {name}")
            selected_keys.update(keys)

    infos = [
        info for info in cache.descriptors
        if selected_keys is None or info["key"] in selected_keys
    ]
    if not infos:
        raise RuntimeError("Aucun descripteur demandé n'est disponible dans l'index.")

    if k <= 0:
        return []

    if candidate_ids is None:
        candidate_indices = np.arange(cache.n)
    else:
        positions = {
            cache.path_to_pos[c]
            for c in (_clean_id(p, dataset) for p in candidate_ids)
            if c in cache.path_to_pos
        }
        candidate_indices = np.array(sorted(positions), dtype=np.intp)
    if candidate_indices.size == 0:
        return []

    fused = np.zeros(candidate_indices.size, dtype=np.float64)
    total_weight = 0.0

    for info in infos:
        query_vec = info["descriptor"].extract(image)
        dists = _compute_distances(query_vec, info)[candidate_indices]

        if NORMALIZATION == "minmax":
            d_min, d_max = dists.min(), dists.max()
            span = d_max - d_min
            if span > 1e-10:
                normed = (dists - d_min) / span
            else:
                normed = np.zeros(candidate_indices.size, dtype=np.float64)
        else:  # "scale" : ~1 = distance typique, 0 = identique
            normed = dists / info["scale"]

        w = info["weight"]
        fused += w * normed
        total_weight += w

    fused /= total_weight  # total_weight > 0 car infos non vide

    # Image seule (pas de filtre texte) : vote de catégorie
    if CATEGORY_RERANK and dataset == "objects" and candidate_ids is None:
        fused = _category_rerank(fused, candidate_indices, cache)

    actual_k = min(k, candidate_indices.size)
    top_idx = np.argpartition(fused, actual_k - 1)[:actual_k]
    sorted_idx = top_idx[np.argsort(fused[top_idx])]

    return [
        {
            "id": cache.all_paths[candidate_indices[i]],
            "distance": round(float(fused[i]), 6),
        }
        for i in sorted_idx
    ]


# ── API publique ──────────────────────────────────────────────────────────────

def search(
    image_path: str,
    descriptor_name: str = "auto",
    distance_name: str = "auto",
    k: int = 10,
    index_file: str = "index.pkl",
) -> list[tuple[str, float]]:
    """
    Recherche à partir d'un fichier image.

    `distance_name` et `index_file` sont conservés pour compatibilité mais
    ignorés : les distances sont fixées par DESCRIPTOR_CONFIG et l'index est
    chargé par load_index().
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    image = Image.open(img_path).convert("RGB")
    results = _optimized_search(image, k, [descriptor_name])
    return [(r["id"], r["distance"]) for r in results]


def visual_search(
    dataset: str,
    query_image_id: str,
    descriptors: list[str],
    k: int = 20,
    candidate_ids: list[str] | set[str] | None = None,
) -> list[dict]:
    """
    Recherche visuelle à partir d'une image déjà dans la base
    (POST /search). `query_image_id` est relatif à data/<dataset>/.
    """
    clean_id = query_image_id.replace("\\", "/")
    if clean_id.startswith(f"{dataset}/"):
        clean_id = clean_id[len(dataset) + 1:]

    base_dir = (DATA_DIR / dataset).resolve()
    img_path = (base_dir / clean_id).resolve()

    # Protection contre le path traversal ("../", chemins absolus, etc.)
    if not img_path.is_relative_to(base_dir):
        raise ValueError(f"Identifiant d'image invalide : {query_image_id}")
    if not img_path.is_file():
        raise FileNotFoundError(f"Image introuvable : {img_path}")

    image = Image.open(img_path).convert("RGB")
    return _optimized_search(image, k, descriptors, candidate_ids, dataset)


def visual_search_image(
    dataset: str,
    image_bytes: bytes,
    descriptors: list[str],
    k: int = 20,
    candidate_ids: list[str] | set[str] | None = None,
) -> list[dict]:
    """Recherche visuelle à partir d'une image uploadée (POST /search/upload)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _optimized_search(image, k, descriptors, candidate_ids, dataset)

    