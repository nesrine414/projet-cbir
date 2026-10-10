"""
Lecture et écriture de l'index des signatures.

Structure de l'index (dict Python) :
{
    "hsv_8_4_4": {
        "data/objects/accordion/image_0001.jpg": np.array([...]),
        "data/objects/accordion/image_0002.jpg": np.array([...]),
        ...
    },
    "lbp_P24_R3": { ... },
    "hog_o9_c8x8_b2x2": { ... }
}

Stockage : fichier pickle dans index/
"""

import pickle
from pathlib import Path
import numpy as np


# Dossier par défaut où sont sauvegardés les index
INDEX_DIR = Path(__file__).resolve().parents[3] / "index"


def save_index(index: dict, filename: str = "index.pkl") -> Path:
    """
    Sauvegarde l'index dans un fichier pickle.

    Args:
        index    : dictionnaire {descripteur_name: {chemin: vecteur}}.
        filename : nom du fichier de sortie (dans index/).

    Returns:
        Chemin absolu du fichier sauvegardé.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    path = INDEX_DIR / filename

    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[store] Index sauvegardé : {path}")
    return path


def load_index(filename: str = "index.pkl") -> dict:
    """
    Charge l'index depuis un fichier pickle.

    Args:
        filename : nom du fichier (dans index/).

    Returns:
        Dictionnaire {descripteur_name: {chemin: vecteur}}.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
    """
    path = INDEX_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"[store] Index introuvable : {path}\n"
            "Lance d'abord : python scripts/build_index.py"
        )

    with open(path, "rb") as f:
        index = pickle.load(f)

    total = sum(len(v) for v in index.values())
    print(f"[store] Index chargé : {len(index)} descripteur(s), {total} signatures")
    return index


def get_descriptor_index(index: dict, descriptor_name: str) -> dict:
    """
    Extrait la sous-partie de l'index pour un descripteur donné.

    Args:
        index           : index complet.
        descriptor_name : ex. 'hsv_8_4_4'.

    Returns:
        {chemin_image: vecteur_numpy}

    Raises:
        KeyError: si le descripteur n'est pas dans l'index.
    """
    if descriptor_name not in index:
        raise KeyError(
            f"[store] Descripteur '{descriptor_name}' absent de l'index.\n"
            f"Disponibles : {list(index.keys())}"
        )
    return index[descriptor_name]
