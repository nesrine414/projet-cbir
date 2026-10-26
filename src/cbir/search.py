"""Image retrieval over the per-dataset descriptor indexes."""

import io
from pathlib import Path

import numpy as np
from PIL import Image

from cbir.descriptors.color import HSVDescriptor
from cbir.descriptors.mpeg7 import (
    ColorLayoutDescriptor,
    EdgeHistogramDescriptor,
    ScalableColorDescriptor,
)
from cbir.descriptors.shape import HOGDescriptor
from cbir.descriptors.texture import LBPDescriptor
from cbir.index.vptree import VPTree
from cbir.index.store import load_index

try:
    from cbir.descriptors.deep import ResNetDescriptor
except (ImportError, OSError, RuntimeError):
    # ResNet is an optional dependency; classic descriptors remain available without it.
    ResNetDescriptor = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_FILES = {"objects": "index.pkl", "mirflickr": "mirflickr.pkl"}
DATASET_DIRS = {
    "objects": DATA_DIR / "objects",
    # The selected source is MIRFLICKR-25K; data/mirflickr holds an older thumbnail set.
    "mirflickr": DATA_DIR / "mirflickr25k",
}

# id, display name, index key, descriptor, distance, fusion weight
DESCRIPTOR_CONFIG = [
    ("color", "HSV color", "hsv_grid3_8_4_4", HSVDescriptor(), "chi2", 0.10),
    ("lbp", "LBP texture", "lbp_grid3_P8_R1", LBPDescriptor(), "chi2", 0.20),
    ("hog", "HOG shape", "hog_o9_c16x16_b2x2", HOGDescriptor(), "l2", 0.70),
    ("cld", "MPEG-7 Color Layout", "cld_6_3_3", ColorLayoutDescriptor(), "l2", 0.1),
    ("ehd", "MPEG-7 Edge Histogram", "ehd_4x4_e5", EdgeHistogramDescriptor(), "l1", 0.1),
    ("scd", "MPEG-7 Scalable Color", "scd_16_4_4", ScalableColorDescriptor(), "l1", 0.1),
]

if ResNetDescriptor is not None:
    DESCRIPTOR_CONFIG.append(
        ("resnet", "Semantic features (ResNet18)", "resnet18_512d", ResNetDescriptor(), "cosine", 8.0)
    )

DESCRIPTOR_ALIASES = {
    "hsv": "color",
    "texture": "lbp",
    "shape": "hog",
    "color-layout": "cld",
    "edge-histogram": "ehd",
    "scalable-color": "scd",
}


class IndexUnavailable(RuntimeError):
    """Raised when a dataset has no built descriptor index."""


class DescriptorUnavailable(ValueError):
    """Raised when a requested descriptor is not present in the index."""


def _dataset_index_file(dataset: str) -> str:
    return INDEX_FILES.get(dataset, f"{dataset}.pkl")


def _relative_id(raw_path: str, dataset: str) -> str | None:
    """Normalize index keys to IDs relative to one dataset's image directory."""
    parts = [part for part in str(raw_path).replace("\\", "/").split("/") if part]
    if parts and parts[0] == "data":
        parts.pop(0)
    if parts and parts[0] in INDEX_FILES:
        if parts[0] != dataset:
            return None
        parts.pop(0)
    return "/".join(parts) if parts else None


def _candidate_id(raw_path: str, dataset: str) -> str:
    parts = [part for part in str(raw_path).replace("\\", "/").split("/") if part]
    if parts and parts[0] == "data":
        parts.pop(0)
        if parts and parts[0] == dataset:
            parts.pop(0)
    return "/".join(parts)


class _IndexCache:
    """Loads each dataset once and keeps its signatures in aligned NumPy matrices."""

    def __init__(self):
        self._datasets: dict[str, dict] = {}

    def ensure_loaded(self, dataset: str = "objects") -> dict:
        if dataset in self._datasets:
            return self._datasets[dataset]

        try:
            raw_index = load_index(_dataset_index_file(dataset))
        except FileNotFoundError:
            raw_index = {}

        available = []
        for descriptor_id, label, key, descriptor, distance, weight in DESCRIPTOR_CONFIG:
            raw_signatures = raw_index.get(key)
            if not raw_signatures:
                continue

            clean_signatures = {}
            for raw_path, vector in raw_signatures.items():
                image_id = _relative_id(raw_path, dataset)
                if image_id is not None:
                    clean_signatures[image_id] = np.asarray(vector, dtype=np.float32).reshape(-1)
            if not clean_signatures:
                continue

            paths = sorted(clean_signatures)
            matrix = np.stack([clean_signatures[path] for path in paths])
            if not np.isfinite(matrix).all():
                raise ValueError(f"Index {key!r} contains NaN or infinite values")
            available.append({
                "id": descriptor_id,
                "label": label,
                "key": key,
                "descriptor": descriptor,
                "dist_type": distance,
                "weight": weight,
                "paths": paths,
                "positions": {path: i for i, path in enumerate(paths)},
                "matrix": matrix,
            })

        state = {
            "descriptors": available,
            "count": max((len(info["paths"]) for info in available), default=0),
            "keys": [info["key"] for info in available],
        }
        self._datasets[dataset] = state
        return state

    def reset(self, dataset: str | None = None) -> None:
        if dataset is None:
            self._datasets.clear()
        else:
            self._datasets.pop(dataset, None)

    # Compatibility views used by the existing evaluation and benchmark scripts.
    @property
    def descriptors(self) -> list[dict]:
        return self.ensure_loaded("objects")["descriptors"]

    @property
    def all_paths(self) -> list[str]:
        descriptors = self.descriptors
        if not descriptors:
            return []
        paths = set(descriptors[0]["paths"])
        for info in descriptors[1:]:
            paths.intersection_update(info["paths"])
        return sorted(paths)

    @property
    def n(self) -> int:
        return len(self.all_paths)

    @property
    def descs(self) -> dict[str, dict]:
        return {info["key"]: info for info in self.descriptors}


_cache = _IndexCache()


def get_index_status(dataset: str = "objects") -> dict:
    """Return the descriptors available to the API for one dataset."""
    state = _cache.ensure_loaded(dataset)
    by_id = {info["id"]: info for info in state["descriptors"]}
    return {
        "indexed": bool(state["descriptors"]),
        "indexed_count": state["count"],
        "descriptors": [
            {
                "id": descriptor_id,
                "label": label,
                "key": key,
                "available": descriptor_id in by_id,
            }
            for descriptor_id, label, key, *_rest in DESCRIPTOR_CONFIG
        ],
    }


def _compute_distances(
    query: np.ndarray,
    info: dict,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    matrix = info["matrix"] if rows is None else info["matrix"][rows]
    distance_type = info["dist_type"]

    if matrix.shape[1] != q.size:
        raise ValueError(
            f"Descriptor {info['key']} has {matrix.shape[1]} indexed values but "
            f"the query produced {q.size}. Rebuild the index with matching descriptor settings."
        )
    if distance_type == "chi2":
        difference = matrix - q
        return 0.5 * np.sum(difference ** 2 / (matrix + q + 1e-10), axis=1)
    if distance_type == "l1":
        return np.sum(np.abs(matrix - q), axis=1)
    if distance_type == "cosine":
        similarities = matrix @ q
        return np.maximum(0.0, 0.5 * (1.0 - similarities))

    q_norm = np.dot(q, q)
    dots = matrix @ q
    squared = np.sum(matrix ** 2, axis=1) + q_norm - 2.0 * dots
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared)


def _scalar_distance(name: str):
    if name == "chi2":
        return lambda left, right: float(0.5 * np.sum((left - right) ** 2 / (left + right + 1e-10)))
    if name == "l1":
        return lambda left, right: float(np.sum(np.abs(left - right)))
    if name == "cosine":
        return lambda left, right: float(max(0.0, 0.5 * (1.0 - np.dot(left, right))))
    return lambda left, right: float(np.linalg.norm(left - right))


def _selected_descriptors(state: dict, requested: list[str] | None) -> list[dict]:
    available = state["descriptors"]
    if not available:
        raise IndexUnavailable("No visual index is available for this dataset. Build it with scripts/build_index.py.")
    if not requested or any(str(name).lower() == "auto" for name in requested):
        selected = available
    else:
        ids = set()
        unknown = []
        for requested_id in requested:
            key = DESCRIPTOR_ALIASES.get(str(requested_id).lower(), str(requested_id).lower())
            matches = [info for info in DESCRIPTOR_CONFIG if info[0] == key or info[2] == key]
            if not matches:
                unknown.append(requested_id)
                continue
            ids.add(matches[0][0])
        if unknown:
            raise DescriptorUnavailable(f"Unknown descriptor(s): {', '.join(map(str, unknown))}")
        selected = [info for info in available if info["id"] in ids]
        unavailable = ids - {info["id"] for info in selected}
        if unavailable:
            labels = [next(spec[1] for spec in DESCRIPTOR_CONFIG if spec[0] == name) for name in unavailable]
            raise DescriptorUnavailable(
                "Not indexed: " + ", ".join(labels) + ". Build the dataset index to enable them."
            )
    if not selected:
        raise IndexUnavailable("No visual index is available for this dataset. Build it with scripts/build_index.py.")
    return selected


def _optimized_search(
    image: Image.Image,
    k: int,
    descriptors: list[str] | None = None,
    candidate_ids: list[str] | set[str] | None = None,
    dataset: str = "objects",
) -> list[dict]:
    """Rank image IDs by the weighted, normalized distance of selected descriptors."""
    state = _cache.ensure_loaded(dataset)
    infos = _selected_descriptors(state, descriptors)

    common_paths = set(infos[0]["paths"])
    for info in infos[1:]:
        common_paths.intersection_update(info["paths"])
    if candidate_ids is not None:
        candidates = {_candidate_id(path, dataset) for path in candidate_ids}
        common_paths.intersection_update(candidates)
    if not common_paths or k <= 0:
        return []

    paths = sorted(common_paths)
    if len(infos) == 1 and infos[0]["id"] != "resnet" and candidate_ids is None and len(paths) <= 12000:
        info = infos[0]
        query = info["descriptor"].extract(image)
        tree = info.get("tree")
        if tree is None:
            signatures = {path: info["matrix"][position] for position, path in enumerate(info["paths"])}
            tree = VPTree(signatures, _scalar_distance(info["dist_type"]))
            info["tree"] = tree
        return [
            {"id": image_id, "distance": round(float(distance), 6)}
            for image_id, distance in tree.search(query, k=min(int(k), len(paths)))
        ]

    total_distance = np.zeros(len(paths), dtype=np.float64)
    weight_sum = 0.0

    for info in infos:
        rows = np.fromiter((info["positions"][path] for path in paths), dtype=np.intp, count=len(paths))
        query = info["descriptor"].extract(image)
        distances = _compute_distances(query, info, rows)
        low, high = float(distances.min()), float(distances.max())
        normalized = (distances - low) / max(high - low, 1e-10)
        weight = info["weight"]
        total_distance += weight * normalized
        weight_sum += weight

    if weight_sum:
        total_distance /= weight_sum

    count = min(int(k), len(paths))
    if count < len(paths):
        selected = np.argpartition(total_distance, count - 1)[:count]
        selected = selected[np.argsort(total_distance[selected], kind="stable")]
    else:
        selected = np.argsort(total_distance, kind="stable")
    return [
        {"id": paths[index], "distance": round(float(total_distance[index]), 6)}
        for index in selected
    ]


def search(
    image_path: str,
    descriptor_name: str = "auto",
    distance_name: str = "auto",
    k: int = 10,
    index_file: str = "index.pkl",
) -> list[tuple[str, float]]:
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path_obj) as source:
        image = source.convert("RGB")
    results = _optimized_search(image, k, [descriptor_name])
    return [(result["id"], result["distance"]) for result in results]


def visual_search(
    dataset: str,
    query_image_id: str,
    descriptors: list[str] | None,
    k: int = 20,
    candidate_ids: list[str] | set[str] | None = None,
) -> list[dict]:
    """Search from an image already stored in the selected dataset."""
    root = DATASET_DIRS.get(dataset, DATA_DIR / dataset).resolve()
    query_path = (root / query_image_id.replace("\\", "/")).resolve()
    if root not in query_path.parents or not query_path.is_file():
        raise FileNotFoundError(f"Image not found in dataset {dataset}: {query_image_id}")
    with Image.open(query_path) as source:
        image = source.convert("RGB")
    return _optimized_search(image, k, descriptors, candidate_ids, dataset)


def visual_search_image(
    dataset: str,
    image_bytes: bytes,
    descriptors: list[str] | None,
    k: int = 20,
    candidate_ids: list[str] | set[str] | None = None,
) -> list[dict]:
    """Search from an uploaded image without storing the upload on disk."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    return _optimized_search(image, k, descriptors, candidate_ids, dataset)
