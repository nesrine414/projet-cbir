import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray

from .base import Descriptor


class HOGDescriptor(Descriptor):
    """
    Descripteur de forme/gradient basé sur le Histogram of Oriented Gradients.

    Principe : l'image est divisée en petites cellules. Dans chaque cellule,
    on calcule la direction et l'intensité des gradients (contours, bords),
    puis on en fait un histogramme. Les cellules sont regroupées en blocs
    pour normaliser l'illumination.

    Le vecteur final = concaténation de tous les histogrammes de cellules.

    Taille fixe grâce au redimensionnement préalable à (128, 128) :
    → avec les params par défaut : 1764 valeurs.
    """

    # Taille fixe pour garantir un vecteur de dimension constante
    TARGET_SIZE = (128, 128)

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
    ):
        """
        Args:
            orientations    : nombre de bins de directions (0°-180°).
            pixels_per_cell : taille de chaque cellule en pixels.
            cells_per_block : nombre de cellules par bloc (pour normalisation).
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    @property
    def name(self) -> str:
        return (
            f"hog_o{self.orientations}"
            f"_c{'x'.join(str(p) for p in self.pixels_per_cell)}"
            f"_b{'x'.join(str(c) for c in self.cells_per_block)}"
        )

    def extract(self, image: Image.Image) -> np.ndarray:
        # 1. Redimensionner à taille fixe — OBLIGATOIRE pour avoir un vecteur
        #    de taille constante quelle que soit l'image d'entrée
        img_resized = image.convert("RGB").resize(self.TARGET_SIZE, Image.LANCZOS)

        # 2. Convertir en niveaux de gris (HOG travaille sur les gradients
        #    d'intensité, pas sur la couleur)
        img_gray = rgb2gray(np.array(img_resized))

        # 3. Calculer le HOG
        # feature_vector=True → retourne directement un vecteur 1D
        features = hog(
            img_gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm="L2-Hys",   # normalisation robuste au contraste
            feature_vector=True,
        )

        # 4. Déjà normalisé par skimage (block_norm="L2-Hys")
        #    On cast juste en float32 pour cohérence avec les autres descripteurs
        return features.astype(np.float32)
