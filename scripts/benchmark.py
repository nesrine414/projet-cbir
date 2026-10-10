"""
Benchmark : recherche exhaustive (linéaire) vs VP-Tree.

Compare les deux approches sur N images requêtes aléatoires :
    - Temps de construction de l'index VP-Tree
    - Temps moyen de recherche
    - Nombre de calculs de distance
    - Exactitude (les résultats sont-ils identiques ?)

Usage :
    python scripts/benchmark.py
    python scripts/benchmark.py --descriptor lbp --n 50 --k 10
"""

import sys
import time
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from cbir.descriptors.color import HSVDescriptor, RGBDescriptor
from cbir.descriptors.texture import LBPDescriptor
from cbir.descriptors.shape import HOGDescriptor
from cbir.distances import get_distance
from cbir.index.store import load_index, get_descriptor_index
from cbir.index import linear
from cbir.index.vptree import VPTree

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "objects"

DESCRIPTORS = {
    "hsv":     HSVDescriptor(),
    "rgb":     RGBDescriptor(),
    "lbp":     LBPDescriptor(),
    "hog":     HOGDescriptor(),
}

DEFAULT_DISTANCE = {
    "hsv": "chi2", "rgb": "chi2",
    "lbp": "chi2",
    "hog": "l2",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_images(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def results_match(r1: list, r2: list, k: int) -> bool:
    """Vérifie que les deux listes de résultats contiennent les mêmes images."""
    ids1 = {item[0] if isinstance(item, tuple) else item["id"] for item in r1[:k]}
    ids2 = {item[0] if isinstance(item, tuple) else item["id"] for item in r2[:k]}
    return ids1 == ids2


def print_separator():
    print("-" * 60)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(descriptor_name: str, n_queries: int, k: int):
    print(f"\n{'='*60}")
    print(f"  BENCHMARK CBIR")
    print(f"  Descripteur : {descriptor_name.upper()}")
    print(f"  Requetes    : {n_queries}  |  k = {k}")
    print(f"{'='*60}")

    # 1. Charger l'index
    print("\n[1/4] Chargement de l'index...")
    index = load_index()
    descriptor = DESCRIPTORS[descriptor_name]
    desc_index = get_descriptor_index(index, descriptor.name)
    distance_fn = get_distance(DEFAULT_DISTANCE.get(descriptor_name, "chi2"))
    n_total = len(desc_index)
    print(f"      {n_total} signatures chargees ({descriptor.name})")

    # 2. Construire le VP-Tree
    print("\n[2/4] Construction du VP-Tree...")
    t0 = time.perf_counter()
    tree = VPTree(desc_index, distance_fn)
    build_time = time.perf_counter() - t0
    print(f"      Temps de construction : {build_time:.2f}s")

    # 3. Choisir N images requetes aleatoires
    print(f"\n[3/4] Selection de {n_queries} images requetes aleatoires...")
    all_images = collect_images(DATA_DIR)
    queries = random.sample(all_images, min(n_queries, len(all_images)))

    # 4. Comparer les deux methodes
    print(f"\n[4/4] Comparaison lineaire vs VP-Tree sur {len(queries)} requetes...\n")
    print_separator()

    linear_times = []
    vptree_times = []
    linear_comps = []
    vptree_comps = []
    exact_matches = 0

    for i, img_path in enumerate(queries, 1):
        img = Image.open(img_path).convert("RGB")
        query_vec = descriptor.extract(img)

        # --- Recherche lineaire ---
        t0 = time.perf_counter()
        res_linear = linear.search(query_vec, desc_index, distance_fn, k=k)
        t_linear = time.perf_counter() - t0
        linear_times.append(t_linear)
        linear_comps.append(n_total)  # toujours n comparaisons

        # --- Recherche VP-Tree ---
        t0 = time.perf_counter()
        res_vptree = tree.search(query_vec, k=k)
        t_vptree = time.perf_counter() - t0
        vptree_times.append(t_vptree)
        vptree_comps.append(tree.n_computations)

        # --- Exactitude ---
        match = results_match(res_linear, res_vptree, k)
        if match:
            exact_matches += 1

        if i % 10 == 0 or i == len(queries):
            print(f"  [{i:3d}/{len(queries)}] "
                  f"Lin: {t_linear*1000:6.1f}ms ({n_total} calc)  |  "
                  f"VPT: {t_vptree*1000:6.1f}ms ({tree.n_computations:4d} calc)  |  "
                  f"{'OK' if match else 'DIFF'}")

    # 5. Resultats
    print_separator()
    avg_linear_ms  = sum(linear_times) / len(linear_times) * 1000
    avg_vptree_ms  = sum(vptree_times) / len(vptree_times) * 1000
    avg_linear_comp = sum(linear_comps) / len(linear_comps)
    avg_vptree_comp = sum(vptree_comps) / len(vptree_comps)
    speedup = avg_linear_ms / avg_vptree_ms if avg_vptree_ms > 0 else float("inf")
    comp_reduction = (1 - avg_vptree_comp / avg_linear_comp) * 100
    accuracy = exact_matches / len(queries) * 100

    print(f"\n{'='*60}")
    print(f"  RESULTATS")
    print(f"{'='*60}")
    print(f"  {'Methode':<25} {'Temps moyen':>12} {'Calculs moyens':>15}")
    print(f"  {'-'*52}")
    print(f"  {'Lineaire (exhaustif)':<25} {avg_linear_ms:>10.2f}ms {avg_linear_comp:>15.0f}")
    print(f"  {'VP-Tree':<25} {avg_vptree_ms:>10.2f}ms {avg_vptree_comp:>15.0f}")
    print(f"  {'-'*52}")
    print(f"  Acceleration (speedup)   : {speedup:.1f}x")
    print(f"  Calculs evites           : {comp_reduction:.1f}%")
    print(f"  Exactitude VP-Tree       : {accuracy:.1f}% ({exact_matches}/{len(queries)} requetes)")
    print(f"  Temps construction arbre : {build_time:.2f}s")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark lineaire vs VP-Tree")
    parser.add_argument("--descriptor", choices=list(DESCRIPTORS.keys()),
                        default="hsv", help="Descripteur a utiliser (defaut: hsv)")
    parser.add_argument("--n", type=int, default=30,
                        help="Nombre de requetes aleatoires (defaut: 30)")
    parser.add_argument("--k", type=int, default=10,
                        help="Nombre de resultats par requete (defaut: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args.descriptor, args.n, args.k)
