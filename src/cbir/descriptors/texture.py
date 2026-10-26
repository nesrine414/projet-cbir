import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

from .base import Descriptor


def _grid_cells(img, n=3):
    """Découpe l'image en n x n cellules régulières."""
    h, w = img.shape[:2]
    for i in range(n):
        for j in range(n):
            yield img[i * h // n : (i + 1) * h // n, j * w // n : (j + 1) * w // n]


class LBPDescriptor(Descriptor):
    """
    Descripteur de texture spatialisé basé sur le Local Binary Pattern (LBP) uniforme en grille 3x3.

    L'image est redimensionnée à 128x128 puis découpée en 3x3 = 9 cellules.
    Pour chaque cellule, un LBP uniforme (P=8, R=1, 10 bins) est calculé et normalisé L1.
    Taille totale : 9 * 10 = 90 valeurs.
    """

    TARGET_SIZE = (128, 128)

    def __init__(self, n_points: int = 8, radius: int = 1, grid: int = 3):
        self.n_points = n_points
        self.radius = radius
        self.grid = grid
        self.n_bins = n_points + 2  # 10 bins pour P=8 uniform

    @property
    def name(self) -> str:
        return f"lbp_grid{self.grid}_P{self.n_points}_R{self.radius}"

    def extract(self, image: Image.Image) -> np.ndarray:
        img_resized = image.convert("L").resize(self.TARGET_SIZE, Image.BILINEAR)
        gray = np.array(img_resized, dtype=np.uint8)

        feats = []
        for cell in _grid_cells(gray, self.grid):
            lbp = local_binary_pattern(cell, P=self.n_points, R=self.radius, method="uniform")
            hist, _ = np.histogram(lbp, bins=self.n_bins, range=(0, self.n_bins))
            total = hist.sum() + 1e-8
            feats.append((hist / total).astype(np.float32))

        return np.concatenate(feats)
