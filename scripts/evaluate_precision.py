"""
Evaluation script — Precision@k (Step 4 of CBIR benchmark).

Mesure la précision moyenne à k (par défaut k=10) pour chaque descripteur seul
et différentes combinaisons de poids.
Produit un tableau comparatif prêt à insérer dans le rapport.

Usage :
    python scripts/evaluate_precision.py
    python scripts/evaluate_precision.py --k 10 --queries 100
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import argparse
from pathlib import Path

# Ajouter src/ au path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image
from tqdm import tqdm

from cbir.search import _cache, _optimized_search


def get_category(path: str) -> str:
    clean = path.replace("\\", "/")
    parts = clean.split("/")
    return parts[0] if len(parts) > 1 else ""


def evaluate_precision(
    k: int = 10,
    n_queries: int = 100,
    seed: int = 42,
) -> dict:
    _cache.ensure_loaded()
    all_paths = _cache.all_paths
    total = len(all_paths)

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(total, size=min(n_queries, total), replace=False)

    configurations = [
        ("Couleur seule (HSV 3x3)", ["color"]),
        ("Texture seule (LBP 3x3)", ["texture"]),
        ("Forme seule (HOG 16x16)", ["shape"]),
        ("Tous (Color 0.35 + Texture 0.25 + Shape 0.40)", ["color", "texture", "shape"]),
    ]

    results = {label: [] for label, _ in configurations}

    data_dir = Path(__file__).resolve().parents[1] / "data" / "objects"

    print(f"\n--- Évaluation sur {len(sampled_indices)} requêtes (k={k}) ---\n")

    for idx in tqdm(sampled_indices, desc="Évaluation Precision@k"):
        rel_path = all_paths[idx]
        query_category = get_category(rel_path)
        img_path = data_dir / rel_path
        if not img_path.exists():
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        for label, descs in configurations:
            hits_list = _optimized_search(image, k=k + 1, descriptors=descs)
            # Retirer la requête elle-même si présente
            hits_filtered = [h for h in hits_list if h["id"] != rel_path][:k]
            correct = sum(
                1 for h in hits_filtered if get_category(h["id"]) == query_category
            )
            prec = correct / k if k > 0 else 0.0
            results[label].append(prec)

    print("\n" + "=" * 65)
    print("📊 TABLEAU COMPARATIF : PRECISION@k")
    print("=" * 65)
    print(f"| {'Configuration':<45} | {'Precision@' + str(k):<12} |")
    print(f"|{'-' * 47}|{'-' * 14}|")
    for label, precs in results.items():
        avg_prec = np.mean(precs) * 100 if precs else 0.0
        print(f"| {label:<45} | {avg_prec:6.2f} %     |")
    print("=" * 65 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation de la Precision@k du moteur CBIR")
    parser.add_argument("--k", type=int, default=10, help="Nombre de résultats évalués (défaut: 10)")
    parser.add_argument("--queries", type=int, default=100, help="Nombre de requêtes d'évaluation (défaut: 100)")
    args = parser.parse_args()

    evaluate_precision(k=args.k, n_queries=args.queries)
