import cv2
import numpy as np
from PIL import Image

from .base import Descriptor


class HSVDescriptor(Descriptor):
    """
    Descripteur couleur basé sur un histogramme HSV 3D.

    Le vecteur final = histogramme aplati et normalisé (L1).
    Taille : bins[0] * bins[1] * bins[2]  →  par défaut 8*4*4 = 128 valeurs.
    """

    def __init__(self, bins: tuple[int, int, int] = (8, 4, 4)):
        """
        Args:
            bins: Nombre de bins pour chaque canal (H, S, V).
        """
        self.bins = bins

    @property
    def name(self) -> str:
        return f"hsv_{'_'.join(str(b) for b in self.bins)}"

    def extract(self, image: Image.Image) -> np.ndarray:
        # 1. PIL (RGB) → numpy → BGR (format OpenCV)
        img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # 2. BGR → HSV
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 3. Histogramme 3D
        # Plages OpenCV : H ∈ [0, 180], S ∈ [0, 256], V ∈ [0, 256]
        hist = cv2.calcHist(
            [img_hsv],
            channels=[0, 1, 2],
            mask=None,
            histSize=list(self.bins),
            ranges=[0, 180, 0, 256, 0, 256],
        )

        # 4. Aplatir et normaliser (L1 : somme = 1)
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total

        return hist


class RGBDescriptor(Descriptor):
    """
    Descripteur couleur basé sur un histogramme RGB 3D.

    Alternative à HSV, moins robuste aux changements d'éclairage
    mais plus simple à interpréter.
    Taille : bins[0] * bins[1] * bins[2]  →  par défaut 8*8*8 = 512 valeurs.
    """

    def __init__(self, bins: tuple[int, int, int] = (8, 8, 8)):
        """
        Args:
            bins: Nombre de bins pour chaque canal (R, G, B).
        """
        self.bins = bins

    @property
    def name(self) -> str:
        return f"rgb_{'_'.join(str(b) for b in self.bins)}"

    def extract(self, image: Image.Image) -> np.ndarray:
        # 1. PIL (RGB) → numpy
        img_rgb = np.array(image.convert("RGB"))

        # 2. Histogramme 3D sur les 3 canaux RGB
        hist = cv2.calcHist(
            [img_rgb],
            channels=[0, 1, 2],
            mask=None,
            histSize=list(self.bins),
            ranges=[0, 256, 0, 256, 0, 256],
        )

        # 3. Aplatir et normaliser (L1)
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total

        return hist
