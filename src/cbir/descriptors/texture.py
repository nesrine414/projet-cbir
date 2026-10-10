import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

from .base import Descriptor


class LBPDescriptor(Descriptor):
    """
    Descripteur de texture basé sur le Local Binary Pattern (LBP) uniforme.

    Principe : pour chaque pixel, on compare ses voisins circulaires à lui-même.
    Chaque comparaison donne 0 ou 1 → on obtient un code binaire par pixel.
    L'histogramme de ces codes = la "texture" de l'image.

    Version 'uniform' : ne garde que les patterns avec au max 2 transitions
    binaires (ex: 00011100), ce qui réduit le bruit et donne l'invariance
    par rotation.

    Taille du vecteur : n_points + 2  →  par défaut 24 + 2 = 26 valeurs.
    """

    def __init__(self, n_points: int = 24, radius: int = 3, n_bins: int = 26):
        """
        Args:
            n_points : nombre de voisins sur le cercle (P).
            radius   : rayon du cercle en pixels (R).
            n_bins   : nombre de bins de l'histogramme.
                       Doit valoir n_points + 2 pour capturer tous les
                       patterns uniformes + 1 bin "non-uniforme".
        """
        self.n_points = n_points
        self.radius = radius
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return f"lbp_P{self.n_points}_R{self.radius}"

    def extract(self, image: Image.Image) -> np.ndarray:
        # 1. Convertir en niveaux de gris (LBP travaille sur 1 seul canal)
        gray = np.array(image.convert("L"), dtype=np.uint8)

        # 2. Calculer le LBP
        # method='uniform' → invariant par rotation, seuls les patterns
        # avec ≤ 2 transitions binaires sont conservés individuellement.
        lbp = local_binary_pattern(gray, P=self.n_points, R=self.radius, method="uniform")

        # 3. Histogramme des codes LBP
        # Les valeurs de lbp vont de 0 à n_points + 1 inclus
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.n_bins,
            range=(0, self.n_bins),
        )

        # 4. Normaliser (L1 : somme = 1)
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total

        return hist
