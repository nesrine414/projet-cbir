import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray

from .base import Descriptor


class HOGDescriptor(Descriptor):
    """
    Descripteur de forme/gradient basé sur le Histogram of Oriented Gradients.

    Principe : l'image est redimensionnée à (128, 128) et convertie en niveaux de gris.
    Avec pixels_per_cell=(16, 16) et cells_per_block=(2, 2), il y a 7x7 blocs de 2x2 cellules
    avec 9 orientations = 7 * 7 * 4 * 9 = 1764 valeurs.
    Cette résolution capture la silhouette globale de l'objet de façon robuste.
    """

    TARGET_SIZE = (128, 128)

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (16, 16),
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

    def extract(self, image: Image.Image) -> np.ndarray:
        img_resized = image.convert("RGB").resize(self.TARGET_SIZE, Image.BILINEAR)
        img_gray = rgb2gray(np.array(img_resized))

        features = hog(
            img_gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        )

        return features.astype(np.float32)
