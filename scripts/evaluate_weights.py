"""
Evaluation avancee du CBIR : Precision@k, mAP@k et recherche des meilleurs poids.

Differences avec evaluate_precision.py :
  - Echantillonnage EQUILIBRE : le meme nombre de requetes par classe
    (sinon les grosses classes faciles dominent la moyenne).
  - La classe BACKGROUND_Google est exclue des requetes.
  - Les requetes sont separees en deux moities : TUNE (pour choisir les poids)
    et TEST (pour rapporter le resultat) -> pas de sur-apprentissage.
  - Grille de poids (pas de 0.1) sur les 3 descripteurs.
  - Les distances sont calculees UNE seule fois par requete (rapide).
  - Baseline aleatoire affichee pour reference.

Usage (depuis la racine du projet) :
    python scripts/evaluate_weights.py
    python scripts/evaluate_weights.py --k 10 --per-class 4
"""

import sys

# pyrefly: ignore[missing-attribute]
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from tqdm import tqdm

from cbir.search import _cache, _compute_distances

# Noms logiques -> cles de l'index (identiques a DESCRIPTOR_CONFIG dans search.py)
KEYS = {
    "color": "hsv_grid3_8_4_4",
    "texture": "lbp_grid3_P8_R1",
    "shape": "hog_o9_c16x16_b2x2",
}
EXCLUDED = {"BACKGROUND_Google"}


def category(path: str) -> str:
    return path.replace("\\", "/").split("/")[0]


def minmax(d: np.ndarray) -> np.ndarray:
    return (d - d.min()) / (d.max() - d.min() + 1e-10)


def main(k: int, per_class: int, seed: int) -> None:
    _cache.ensure_loaded()
    paths = _cache.all_paths
    cats = np.array([category(p) for p in paths])
    n_total = len(paths)
    class_size = Counter(cats.tolist())

    by_key = {info["key"]: info for info in _cache.descriptors}
    missing = [key for key in KEYS.values() if key not in by_key]
    if missing:
        raise SystemExit(f"Descripteurs absents de l'index : {missing}. Relance build_index.py")

    # ── Echantillonnage equilibre, split tune / test ─────────────────────────
    rng = np.random.default_rng(seed)
    tune_q, test_q = [], []
    for c in sorted(class_size):
        if c in EXCLUDED:
            continue
        idx = np.where(cats == c)[0]
        if len(idx) < 3:
            continue
        pick = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        half = len(pick) // 2
        tune_q += pick[:half].tolist()
        test_q += pick[half:].tolist()

    print(f"\nImages indexees : {n_total} | classes : {len(class_size)}")
    print(f"Requetes TUNE : {len(tune_q)} | requetes TEST : {len(test_q)} | k = {k}\n")

    # ── Pre-calcul des distances normalisees (une fois par requete) ──────────
    def precompute(queries):
        out = {}
        for qi in tqdm(queries, desc="Distances", leave=False):
            out[qi] = {}
            for name, key in KEYS.items():
                info = by_key[key]
                d = _compute_distances(info["matrix"][qi], info)
                out[qi][name] = minmax(d).astype(np.float32)
        return out

    nd_tune = precompute(tune_q)
    nd_test = precompute(test_q)

    # ── Evaluation d'une configuration de poids ──────────────────────────────
    def evaluate(queries, nd, w):
        precs, aps = [], []
        for qi in queries:
            total = sum(w[name] * nd[qi][name] for name in KEYS)
            total = np.array(total, dtype=np.float64)
            total[qi] = np.inf  # on retire la requete elle-meme
            top = np.argpartition(total, k)[:k]
            top = top[np.argsort(total[top])]
            rel = (cats[top] == cats[qi]).astype(np.float64)
            precs.append(rel.mean())
            n_relevant = min(k, class_size[cats[qi]] - 1)
            ap = ((np.cumsum(rel) / np.arange(1, k + 1)) * rel).sum() / max(n_relevant, 1)
            aps.append(ap)
        return float(np.mean(precs)), float(np.mean(aps))

    # ── Grille de poids (somme = 1, pas de 0.1) ──────────────────────────────
    grid = [
        {"color": a / 10, "texture": b / 10, "shape": (10 - a - b) / 10}
        for a in range(11)
        for b in range(11 - a)
    ]
    tune_scores = []
    for w in tqdm(grid, desc="Grille de poids", leave=False):
        p, _ = evaluate(tune_q, nd_tune, w)
        tune_scores.append(p)
    order = np.argsort(tune_scores)[::-1]
    best_w = grid[order[0]]

    # ── Resultats sur TEST ───────────────────────────────────────────────────
    current_w = {name: by_key[key]["weight"] for name, key in KEYS.items()}
    s = sum(current_w.values())
    current_w = {n: v / s for n, v in current_w.items()}

    rows = [
        ("Couleur seule (HSV grille 3x3)", {"color": 1.0, "texture": 0.0, "shape": 0.0}),
        ("Texture seule (LBP grille 3x3)", {"color": 0.0, "texture": 1.0, "shape": 0.0}),
        ("Forme seule (HOG 16x16)", {"color": 0.0, "texture": 0.0, "shape": 1.0}),
        ("Fusion poids actuels (config)", current_w),
        ("Fusion poids egaux (1/3 chacun)", {n: 1 / 3 for n in KEYS}),
        ("Fusion MEILLEURS poids (choisis sur TUNE)", best_w),
    ]

    random_baseline = np.mean(
        [(class_size[cats[qi]] - 1) / (n_total - 1) for qi in test_q]
    )

    print("=" * 86)
    print(f"RESULTATS SUR LES REQUETES TEST (k={k}, {len(test_q)} requetes, 1 a {per_class // 2 or 1} par classe)")
    print("=" * 86)
    print(f"| {'Configuration':<44} | {'C/T/S':<13} | {'P@' + str(k):>7} | {'mAP@' + str(k):>7} |")
    print(f"|{'-' * 46}|{'-' * 15}|{'-' * 9}|{'-' * 9}|")
    for label, w in rows:
        p, ap = evaluate(test_q, nd_test, w)
        ws = f"{w['color']:.1f}/{w['texture']:.1f}/{w['shape']:.1f}"
        print(f"| {label:<44} | {ws:<13} | {p * 100:6.2f}% | {ap * 100:6.2f}% |")
    print(f"| {'Baseline aleatoire':<44} | {'-':<13} | {random_baseline * 100:6.2f}% | {'-':>7} |")
    print("=" * 86)

    print("\nTop 5 des poids sur TUNE (color/texture/shape -> P@k) :")
    for i in order[:5]:
        w = grid[i]
        print(f"  {w['color']:.1f} / {w['texture']:.1f} / {w['shape']:.1f}  ->  {tune_scores[i] * 100:.2f}%")

    # ── Precision par classe (poids optimaux, TEST) ──────────────────────────
    per_cat = {}
    for qi in test_q:
        total = sum(best_w[name] * nd_test[qi][name] for name in KEYS)
        total = np.array(total, dtype=np.float64)
        total[qi] = np.inf
        top = np.argpartition(total, k)[:k]
        per_cat.setdefault(cats[qi], []).append((cats[top] == cats[qi]).mean())
    ranked = sorted(((np.mean(v), c) for c, v in per_cat.items()), reverse=True)

    print("\nClasses les plus faciles (poids optimaux) :")
    for p, c in ranked[:8]:
        print(f"  {c:<22} {p * 100:5.1f}%")
    print("\nClasses les plus difficiles :")
    for p, c in ranked[-8:]:
        print(f"  {c:<22} {p * 100:5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation CBIR : P@k, mAP et poids optimaux")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--per-class", type=int, default=4,
                        help="Requetes par classe (moitie TUNE, moitie TEST)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.k, args.per_class, args.seed)