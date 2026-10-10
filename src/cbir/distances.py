import numpy as np


def chi2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distance Chi-Carré (χ²) entre deux histogrammes.

    Formule : Σ (a_i - b_i)² / (a_i + b_i + ε)

    - Idéale pour comparer des histogrammes normalisés (HSV, LBP).
    - Très sensible aux petites différences dans les bins dominants.
    - Toujours >= 0. Vaut 0 si et seulement si a == b.

    Args:
        a, b: vecteurs numpy 1D de même taille (histogrammes normalisés).

    Returns:
        Distance scalaire (float).
    """
    eps = 1e-10  # éviter la division par zéro
    return float(np.sum((a - b) ** 2 / (a + b + eps)))


def l1(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distance L1 — Manhattan.

    Formule : Σ |a_i - b_i|

    - Simple et rapide.
    - Moins sensible aux valeurs extrêmes que L2.
    - Toujours >= 0. Vaut 0 si et seulement si a == b.

    Args:
        a, b: vecteurs numpy 1D de même taille.

    Returns:
        Distance scalaire (float).
    """
    return float(np.sum(np.abs(a - b)))


def l2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distance L2 — Euclidienne.

    Formule : √( Σ (a_i - b_i)² )

    - La plus classique.
    - Adaptée pour HOG (vecteurs à composantes continues).
    - Toujours >= 0. Vaut 0 si et seulement si a == b.

    Args:
        a, b: vecteurs numpy 1D de même taille.

    Returns:
        Distance scalaire (float).
    """
    return float(np.sqrt(np.sum((a - b) ** 2)))


# Registre des distances disponibles (utile pour search.py et l'API)
DISTANCES = {
    "chi2": chi2,
    "l1": l1,
    "l2": l2,
}


def get_distance(name: str):
    """
    Retourne la fonction de distance par son nom.

    Args:
        name: 'chi2', 'l1' ou 'l2'.

    Returns:
        Fonction de distance.

    Raises:
        ValueError: si le nom est inconnu.
    """
    if name not in DISTANCES:
        raise ValueError(f"Distance inconnue '{name}'. Choisir parmi : {list(DISTANCES.keys())}")
    return DISTANCES[name]
