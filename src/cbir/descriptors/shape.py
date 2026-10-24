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
    (qui se chevauchent, pas d'une cellule) pour normaliser l'illumination.

    Taille fixe grâce au redimensionnement préalable à (128, 128) :
      128/8 = 16 cellules par côté -> 15 x 15 blocs 2x2
      -> 15 * 15 * (2 * 2 * 9) = 8100 valeurs avec les paramètres par défaut.
    (1764 correspondrait à une image 64x64 : 7*7*36.)
    """

    # Taille fixe pour garantir un vecteur de dimension constante
    TARGET_SIZE = (128, 128)

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
    ):
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

    @property
    def dim(self) -> int:
        """Dimension théorique du vecteur (utile pour vérifier l'index)."""
        h, w = self.TARGET_SIZE
        cy, cx = self.pixels_per_cell
        by, bx = self.cells_per_block
        blocks_y = h // cy - by + 1
        blocks_x = w // cx - bx + 1
        return blocks_y * blocks_x * by * bx * self.orientations

    def extract(self, image: Image.Image) -> np.ndarray:
        # 1. Redimensionner à taille fixe (déforme les images non carrées,
        #    mais garantit un vecteur de taille constante)
        img_resized = image.convert("RGB").resize(self.TARGET_SIZE, Image.LANCZOS)

        # 2. Niveaux de gris : HOG travaille sur les gradients d'intensité
        img_gray = rgb2gray(np.array(img_resized))

        # 3. HOG, vecteur 1D, normalisation L2-Hys par bloc
        features = hog(
            img_gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        )

        return features.astype(np.float32)