"""Module de chargement et gestion des données d'images pour CBIR."""
from pathlib import Path
from typing import List, Tuple, Union

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}


def list_images(
    root: Union[str, Path]
) -> Tuple[List[Path], List[str], List[str]]:
    """Parcourt un dossier d'images et retourne les fichiers, identifiants et catégories.

    Args:
        root: Chemin vers le dossier racine du dataset (ex: 'data/objects').

    Returns:
        files: Liste des objets Path de chaque image trouvée.
        ids: Liste des identifiants relatifs (ex: 'accordion/image_0001.jpg').
        cats: Liste des catégories correspondantes (ex: 'accordion').
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Le dossier spécifié n'existe pas : {root_path}")

    files: List[Path] = []
    ids: List[str] = []
    cats: List[str] = []

    for p in sorted(root_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
            rel_path = p.relative_to(root_path).as_posix()
            ids.append(rel_path)
            # La catégorie est le nom du dossier parent immédiat (ou le premier dossier relatif)
            parts = p.relative_to(root_path).parts
            if len(parts) > 1:
                cat = parts[0]
            else:
                cat = ""
            cats.append(cat)

    return files, ids, cats
