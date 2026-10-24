"""
Évalue objectivement la qualité de la recherche visuelle (leave-one-out).

Principe : pour chaque image requête échantillonnée, on cherche dans l'index
(en s'excluant elle-même) et on mesure quelle fraction des k premiers résultats
est de la MÊME catégorie (1er dossier de l'id, ex. 'butterfly/image_0003.jpg').

Lancement depuis la racine du projet :
    python scripts/evaluate.py
    python scripts/evaluate.py --queries 400 --k 10
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbir.search import _cache, _compute_distances  # noqa: E402


def precision_at_k(fused: np.ndarray, q_idx: np.ndarray, labels: np.ndarray, k: int) -> float:
    """fused : (Q, N) scores (petit = proche). Exclut l'image requête elle-même."""
    fused = fused.copy()
    fused[np.arange(len(q_idx)), q_idx] = np.inf
    top = np.argpartition(fused, k, axis=1)[:, :k]
    same = labels[top] == labels[q_idx][:, None]
    return float(same.mean())


def category_rerank_report(fused, q_idx, labels, k):
    """
    Teste le re-ranking par vote de catégorie (cf. search.py) :
    les K plus proches voisines (hors requête) votent, puis
    score' = score * (1 - beta * part_de_votes_de_la_catégorie).
    """
    cats, cat_idx = np.unique(labels, return_inverse=True)
    n_cats = len(cats)
    base = fused.copy()
    base[np.arange(len(q_idx)), q_idx] = np.inf

    print(f"\n=== Re-ranking par vote de catégorie (Précision@{k}) ===")
    print(f"  sans re-ranking : {precision_at_k(fused, q_idx, labels, k):.3f}")
    best = (0.0, None)
    for K in (5, 7, 10, 15):
        top = np.argpartition(base, K, axis=1)[:, :K]
        shares = np.stack([np.bincount(cat_idx[row], minlength=n_cats) / K for row in top])
        vote_acc = float((shares.argmax(axis=1) == cat_idx[q_idx]).mean())
        line = [f"  K={K:2d} (le vote trouve la bonne catégorie dans {vote_acc:.0%} des cas) :"]
        for beta in (0.7, 0.9, 0.95):
            adj = fused * (1.0 - beta * shares[:, cat_idx])
            p = precision_at_k(adj, q_idx, labels, k)
            line.append(f"beta={beta}: {p:.3f}")
            if p > best[0]:
                best = (p, (K, beta))
        print("  ".join(line))
    print(f"  -> meilleur : P@{k} = {best[0]:.3f} avec CATEGORY_VOTE_K={best[1][0]}, "
          f"CATEGORY_BETA={best[1][1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _cache.ensure_loaded()
    if not _cache.descriptors:
        sys.exit("Index vide : lance d'abord scripts/build_index.py")

    paths = _cache.all_paths
    labels = np.array([p.split("/")[0] for p in paths])
    n = len(paths)
    cats, counts = np.unique(labels, return_counts=True)
    chance = float(((counts / n) ** 2).sum())
    print(f"{n} images, {len(cats)} catégories. Hasard ≈ {chance:.3f} de précision.\n")

    rng = np.random.default_rng(args.seed)
    q_idx = rng.choice(n, size=min(args.queries, n), replace=False)

    # Distances brutes (Q, N) par descripteur, à partir des vecteurs de l'index
    raw, scaled, minmax = {}, {}, {}
    for info in _cache.descriptors:
        D = np.stack([_compute_distances(info["matrix"][i], info) for i in q_idx])
        raw[info["key"]] = D
        scaled[info["key"]] = D / info["scale"]
        mn = D.min(axis=1, keepdims=True)
        span = np.maximum(D.max(axis=1, keepdims=True) - mn, 1e-10)
        minmax[info["key"]] = (D - mn) / span

    keys = list(raw)
    k = args.k

    print(f"=== Précision@{k} par descripteur seul ===")
    for key in keys:
        print(f"  {key:26s} {precision_at_k(raw[key], q_idx, labels, k):.3f}")

    # Configuration actuelle (poids du code)
    current_w = {i["key"]: i["weight"] for i in _cache.descriptors}

    def fuse(norm, w):
        s = sum(w[key] * norm[key] for key in keys if w.get(key, 0) > 0)
        return s / sum(v for v in w.values() if v > 0)

    print(f"\n=== Configuration actuelle {current_w} ===")
    print(f"  min-max (ancien) : {precision_at_k(fuse(minmax, current_w), q_idx, labels, k):.3f}")
    print(f"  scale   (nouveau): {precision_at_k(fuse(scaled, current_w), q_idx, labels, k):.3f}")

    # Recherche des meilleurs poids (normalisation 'scale')
    results = []
    if 3 ** len(keys) <= 800:
        # Peu de descripteurs : grille complète {0, 1, 2}
        for combo in itertools.product([0, 1, 2], repeat=len(keys)):
            if sum(combo) == 0:
                continue
            w = dict(zip(keys, combo))
            results.append((precision_at_k(fuse(scaled, w), q_idx, labels, k), w))
        results.sort(key=lambda t: t[0], reverse=True)
        best_list = results[:8]
        title = "Top 8 des combinaisons de poids"
    else:
        # Beaucoup de descripteurs : montée par coordonnée (2 passes)
        grid = [0, 0.5, 1, 2, 3]
        best_single = max(keys, key=lambda kk: precision_at_k(raw[kk], q_idx, labels, k))
        w = {kk: 0.0 for kk in keys}
        w[best_single] = 1.0
        best_p = precision_at_k(fuse(scaled, w), q_idx, labels, k)
        for _ in range(2):
            for kk in keys:
                for val in grid:
                    trial = dict(w, **{kk: val})
                    if sum(trial.values()) == 0:
                        continue
                    p = precision_at_k(fuse(scaled, trial), q_idx, labels, k)
                    if p > best_p + 1e-9:
                        best_p, w = p, trial
        best_list = [(best_p, w)]
        title = "Meilleurs poids trouvés (recherche par coordonnée)"

    print(f"\n=== {title}, Précision@{k} ===")
    for p, w in best_list:
        print(f"  {p:.3f}  {w}")

    # Re-ranking par catégorie, avec les poids actuellement dans search.py
    category_rerank_report(fuse(scaled, current_w), q_idx, labels, k)


if __name__ == "__main__":
    main()