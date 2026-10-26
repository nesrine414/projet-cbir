import cv2
import numpy as np
from PIL import Image

from .base import Descriptor


def _grid_cells(img, n=3):
    """Découpe l'image en n x n cellules régulières."""
    h, w = img.shape[:2]
    for i in range(n):
        for j in range(n):
            yield img[i * h // n : (i + 1) * h // n, j * w // n : (j + 1) * w // n]


class HSVDescriptor(Descriptor):
    """
    Descripteur couleur spatialisé basé sur un histogramme HSV 3D en grille 3x3.

    L'image est redimensionnée à 128x128 puis découpée en 3x3 = 9 cellules.
    Chaque cellule produit un histogramme 8*4*4 = 128 bins normalisé L1.
    Taille totale du vecteur : 9 * 128 = 1152 valeurs.
    Préserve la localisation spatiale des couleurs (haut = ciel/fond, centre = objet).
    """

    TARGET_SIZE = (128, 128)

    def __init__(self, bins: tuple[int, int, int] = (8, 4, 4), grid: int = 3):
        self.bins = bins
        self.grid = grid

    @property
    def name(self) -> str:
        return f"hsv_grid{self.grid}_{'_'.join(str(b) for b in self.bins)}"

    def extract(self, image: Image.Image) -> np.ndarray:
        img_resized = image.convert("RGB").resize(self.TARGET_SIZE, Image.BILINEAR)
        img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        feats = []
        for cell in _grid_cells(img_hsv, self.grid):
            hist = cv2.calcHist(
                [cell],
                channels=[0, 1, 2],
                mask=None,
                histSize=list(self.bins),
                ranges=[0, 180, 0, 256, 0, 256],
            ).flatten()
            total = hist.sum() + 1e-8
            feats.append((hist / total).astype(np.float32))

        return np.concatenate(feats)


class RGBDescriptor(Descriptor):
    """
    Descripteur couleur spatialisé basé sur un histogramme RGB 3D en grille 3x3.
    """

    TARGET_SIZE = (128, 128)

    def __init__(self, bins: tuple[int, int, int] = (4, 4, 4), grid: int = 3):
        self.bins = bins
        self.grid = grid

    @property
    def name(self) -> str:
        return f"rgb_grid{self.grid}_{'_'.join(str(b) for b in self.bins)}"

    def extract(self, image: Image.Image) -> np.ndarray:
        img_resized = image.convert("RGB").resize(self.TARGET_SIZE, Image.BILINEAR)
        img_rgb = np.array(img_resized)

        feats = []
        for cell in _grid_cells(img_rgb, self.grid):
            hist = cv2.calcHist(
                [cell],
                channels=[0, 1, 2],
                mask=None,
                histSize=list(self.bins),
                ranges=[0, 256, 0, 256, 0, 256],
            ).flatten()
            total = hist.sum() + 1e-8
            feats.append((hist / total).astype(np.float32))

        return np.concatenate(feats)
