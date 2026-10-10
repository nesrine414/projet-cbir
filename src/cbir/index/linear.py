"""
Recherche exhaustive (linéaire) par similarité.

Principe : comparer la signature de la requête avec TOUTES les signatures
de l'index, trier par distance croissante, retourner les k meilleurs.

C'est la référence de correction et le point de comparaison pour le VP-Tree.
Complexité : O(n) par requête (n = nombre d'images dans l'index).
"""

import numpy as np
from typing import Callable


def search(
    query_vector: np.ndarray,
    index: dict,
    distance_fn: Callable,
    k: int = 10,
) -> list[tuple[str, float]]:
    """
    Recherche exhaustive : compare la requête à toutes les signatures.

    Args:
        query_vector : vecteur numpy 1D de la requête (déjà extrait).
        index        : {chemin_image: vecteur_numpy} pour UN descripteur.
        distance_fn  : fonction de distance (chi2, l1, l2).
        k            : nombre de résultats à retourner.

    Returns:
        Liste triée de (chemin_image, distance), du plus similaire au moins.
    """
    scores = []

    for path, vector in index.items():
        dist = distance_fn(query_vector, vector)
        scores.append((path, dist))

    # Trier par distance croissante (distance 0 = identique)
    scores.sort(key=lambda x: x[1])

    return scores[:k]
