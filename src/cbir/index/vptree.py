"""
VP-Tree (Vantage Point Tree) — recherche optimisée par similarité.

Principe :
    On choisit un point "vantage point" (VP), on calcule sa distance
    à tous les autres points, on trouve la médiane µ, puis on divise
    en deux groupes :
        - inner : points à distance ≤ µ du VP
        - outer : points à distance > µ du VP
    On recommence récursivement sur chaque groupe.

Recherche :
    Pour une requête q avec seuil τ (meilleure distance trouvée) :
        - Si d(q, VP) - τ ≤ µ  : explorer inner
        - Si d(q, VP) + τ ≥ µ  : explorer outer
    On "prune" (ignore) les branches qui ne peuvent pas améliorer τ.

Complexité moyenne : O(log n) vs O(n) pour la recherche linéaire.
"""

import random
import numpy as np
from typing import Callable


class _Node:
    """Noeud interne du VP-Tree."""
    __slots__ = ("path", "vector", "mu", "inner", "outer")

    def __init__(self, path, vector, mu=None, inner=None, outer=None):
        self.path = path      # chemin de l'image (vantage point)
        self.vector = vector  # vecteur numpy du vantage point
        self.mu = mu          # médiane des distances (seuil de partition)
        self.inner = inner    # sous-arbre des points proches
        self.outer = outer    # sous-arbre des points éloignés


class VPTree:
    """
    VP-Tree construit sur un index {chemin: vecteur}.

    Usage :
        tree = VPTree(index, distance_fn)
        results = tree.search(query_vector, k=10)
    """

    def __init__(self, index: dict, distance_fn: Callable):
        """
        Construit le VP-Tree.

        Args:
            index       : {chemin_image: vecteur_numpy}.
            distance_fn : fonction de distance (chi2, l1, l2).
        """
        self.distance_fn = distance_fn
        self._n_computations = 0  # compteur pour le benchmark

        # Convertir l'index en liste de (chemin, vecteur) pour la construction
        items = list(index.items())
        self._root = self._build(items)

    # ── Construction ─────────────────────────────────────────────────────────

    def _build(self, items: list) -> _Node | None:
        """Construit récursivement le VP-Tree."""
        if not items:
            return None

        if len(items) == 1:
            path, vec = items[0]
            return _Node(path, vec)

        # 1. Choisir un vantage point aléatoirement
        vp_idx = random.randint(0, len(items) - 1)
        vp_path, vp_vec = items[vp_idx]
        rest = items[:vp_idx] + items[vp_idx + 1:]

        # 2. Calculer les distances de tous les autres points au VP
        distances = [(self.distance_fn(vp_vec, vec), path, vec)
                     for path, vec in rest]

        # 3. Médiane des distances = seuil de partition
        mu = np.median([d for d, _, _ in distances])

        # 4. Partitionner inner (≤ µ) et outer (> µ)
        inner_items = [(p, v) for d, p, v in distances if d <= mu]
        outer_items = [(p, v) for d, p, v in distances if d > mu]

        # 5. Construire récursivement
        return _Node(
            path=vp_path,
            vector=vp_vec,
            mu=mu,
            inner=self._build(inner_items),
            outer=self._build(outer_items),
        )

    # ── Recherche ─────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """
        Recherche les k plus proches voisins dans le VP-Tree.

        Args:
            query_vector : vecteur numpy 1D de la requête.
            k            : nombre de résultats.

        Returns:
            Liste triée de (chemin_image, distance).
        """
        self._n_computations = 0

        # Tas des k meilleurs résultats (max-heap simulé avec liste triée)
        # On garde les k plus petites distances
        heap = []

        self._search_recursive(self._root, query_vector, k, heap)

        # Trier par distance croissante
        heap.sort(key=lambda x: x[1])
        return heap

    def _search_recursive(self, node: _Node | None, query: np.ndarray,
                          k: int, heap: list) -> float:
        """
        Parcours récursif avec pruning.

        Returns:
            τ : meilleure distance courante (seuil de pruning).
        """
        if node is None:
            return float("inf")

        # Distance query → vantage point courant
        d = self.distance_fn(query, node.vector)
        self._n_computations += 1

        # Ajouter ce point aux résultats si bon
        heap.append((node.path, d))
        heap.sort(key=lambda x: x[1])
        if len(heap) > k:
            heap.pop()  # retirer le plus éloigné

        # τ = pire des k meilleures distances actuelles
        tau = heap[-1][1] if len(heap) == k else float("inf")

        if node.mu is None:
            return tau  # feuille

        # Pruning : décider quels sous-arbres explorer
        if d - tau <= node.mu:
            tau = self._search_recursive(node.inner, query, k, heap)
            tau = heap[-1][1] if len(heap) == k else float("inf")

        if d + tau >= node.mu:
            tau = self._search_recursive(node.outer, query, k, heap)

        return heap[-1][1] if len(heap) == k else float("inf")

    @property
    def n_computations(self) -> int:
        """Nombre de calculs de distance lors de la dernière recherche."""
        return self._n_computations
