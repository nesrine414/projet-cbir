"""
Descripteurs spatiaux : mêmes histogrammes que color.py / texture.py, mais
calculés par cellule d'une grille (ex. 3x3) puis concaténés.

Pourquoi : un histogramme global ignore où sont les couleurs/textures. Sur
une base d'objets (Caltech-like), le fond domine l'histogramme global et on
retrouve des images au fond semblable plutôt que le même objet. Avec une
grille, la cellule centrale (l'objet) est comparée à la cellule centrale.
"""

import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

from .base import Descriptor

_SIZE = 192  # divisible par 2, 3, 4 : cellules de taille égale


def _cells(h: int, w: int, gy: int, gx: int):
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)
    for i in range(gy):
        for j in range(gx):
            yield i, j, slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1])


def _finalize(blocks: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Pondère chaque cellule puis normalise L1 globalement (somme = 1)."""
    vec = np.concatenate([b * w for b, w in zip(blocks, weights)]).astype(np.float32)
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


class HSVGridDescriptor(Descriptor):
    """Histogramme HSV par cellule d'une grille. Taille : gy*gx*bins[0]*bins[1]*bins[2]."""

    def __init__(self, grid=(3, 3), bins=(8, 3, 3), center_weight: float = 1.0):
        self.grid = grid
        self.bins = bins
        self.center_weight = center_weight  # >1 : donne plus de poids à la cellule centrale

    @property
    def name(self) -> str:
        return f"hsv_grid_{self.grid[0]}x{self.grid[1]}_{'_'.join(map(str, self.bins))}"

    def extract(self, image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB").resize((_SIZE, _SIZE), Image.LANCZOS))
        hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        gy, gx = self.grid

        blocks, weights = [], []
        for i, j, ys, xs in _cells(_SIZE, _SIZE, gy, gx):
            cell = np.ascontiguousarray(hsv[ys, xs])
            h = cv2.calcHist([cell], [0, 1, 2], None, list(self.bins),
                             [0, 180, 0, 256, 0, 256]).flatten().astype(np.float32)
            s = h.sum()
            if s > 0:
                h /= s
            blocks.append(h)
            is_center = (gy % 2 == 1 and gx % 2 == 1 and i == gy // 2 and j == gx // 2)
            weights.append(self.center_weight if is_center else 1.0)
        return _finalize(blocks, weights)


class LBPGridDescriptor(Descriptor):
    """Histogramme LBP uniforme par cellule. Taille : gy*gx*(n_points+2)."""

    def __init__(self, grid=(3, 3), n_points: int = 24, radius: int = 3,
                 center_weight: float = 1.0):
        self.grid = grid
        self.n_points = n_points
        self.radius = radius
        self.center_weight = center_weight

    @property
    def name(self) -> str:
        return f"lbp_grid_{self.grid[0]}x{self.grid[1]}_P{self.n_points}_R{self.radius}"

    def extract(self, image: Image.Image) -> np.ndarray:
        gray = np.array(image.convert("L").resize((_SIZE, _SIZE), Image.LANCZOS), dtype=np.uint8)
        # LBP calculé une seule fois sur toute l'image (pas d'effet de bord par cellule)
        lbp = local_binary_pattern(gray, P=self.n_points, R=self.radius, method="uniform")
        n_bins = self.n_points + 2
        gy, gx = self.grid

        blocks, weights = [], []
        for i, j, ys, xs in _cells(_SIZE, _SIZE, gy, gx):
            h, _ = np.histogram(lbp[ys, xs].ravel(), bins=n_bins, range=(0, n_bins))
            h = h.astype(np.float32)
            s = h.sum()
            if s > 0:
                h /= s
            blocks.append(h)
            is_center = (gy % 2 == 1 and gx % 2 == 1 and i == gy // 2 and j == gx // 2)
            weights.append(self.center_weight if is_center else 1.0)
        return _finalize(blocks, weights)