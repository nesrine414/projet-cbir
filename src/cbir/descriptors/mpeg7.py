"""
Descripteurs MPEG-7 calculés directement depuis les images (pas besoin des
fichiers de signatures fournis avec la base MirFlickr).

  - ColorLayoutDescriptor   (CLD) : disposition spatiale des couleurs
  - EdgeHistogramDescriptor (EHD) : répartition spatiale des contours
  - ScalableColorDescriptor (SCD) : histogramme HSV + transformée de Haar

Implémentation fidèle au principe de la norme (ISO/IEC 15938-3) mais en
virgule flottante : la quantification normative des coefficients n'est pas
appliquée. Les résultats ne sont donc pas identiques bit à bit aux signatures
du logiciel de référence (XM), mais le comportement est le même.

Distances à utiliser dans search.py :
  CLD -> "l2"  (les poids de la norme sont déjà intégrés dans le vecteur)
  EHD -> "l1"
  SCD -> "l1"
"""

import cv2
import numpy as np
from PIL import Image

from .base import Descriptor


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _zigzag_order(n: int = 8) -> list[tuple[int, int]]:
    """Ordre de parcours en zigzag d'une matrice n x n (comme JPEG)."""
    order = []
    for s in range(2 * n - 1):
        idx = [(i, s - i) for i in range(max(0, s - n + 1), min(s, n - 1) + 1)]
        if s % 2 == 0:
            idx.reverse()
        order.extend(idx)
    return order


_ZIGZAG = _zigzag_order(8)


# ── 1. Color Layout Descriptor ────────────────────────────────────────────────

class ColorLayoutDescriptor(Descriptor):
    """
    CLD : l'image est réduite à une grille 8x8 de couleurs moyennes, passée en
    YCbCr, puis une DCT 8x8 est appliquée à chaque canal. On garde les premiers
    coefficients en zigzag : 6 pour Y, 3 pour Cb, 3 pour Cr -> 12 valeurs.

    Les poids de la distance normative (Y : 2,2,2,1,1,1 ; Cb : 2,1,1 ;
    Cr : 4,2,2) sont intégrés en multipliant chaque coefficient par la racine
    du poids : une simple distance L2 donne alors la distance pondérée.
    """

    N_Y, N_CB, N_CR = 6, 3, 3
    _W_Y = np.array([2, 2, 2, 1, 1, 1], dtype=np.float32)
    _W_CB = np.array([2, 1, 1], dtype=np.float32)
    _W_CR = np.array([4, 2, 2], dtype=np.float32)

    @property
    def name(self) -> str:
        return f"cld_{self.N_Y}_{self.N_CB}_{self.N_CR}"

    @staticmethod
    def _dct_zigzag(channel: np.ndarray, n: int) -> np.ndarray:
        dct = cv2.dct(channel.astype(np.float32))
        return np.array([dct[i, j] for i, j in _ZIGZAG[:n]], dtype=np.float32)

    def extract(self, image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB"))
        # Couleur moyenne de chacun des 64 blocs (INTER_AREA = moyenne par bloc)
        small = cv2.resize(rgb, (8, 8), interpolation=cv2.INTER_AREA)
        # OpenCV renvoie l'ordre Y, Cr, Cb
        ycc = cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]

        vy = self._dct_zigzag(y, self.N_Y) * np.sqrt(self._W_Y)
        vcb = self._dct_zigzag(cb, self.N_CB) * np.sqrt(self._W_CB)
        vcr = self._dct_zigzag(cr, self.N_CR) * np.sqrt(self._W_CR)
        return np.concatenate([vy, vcb, vcr]).astype(np.float32)


# ── 2. Edge Histogram Descriptor ──────────────────────────────────────────────

class EdgeHistogramDescriptor(Descriptor):
    """
    EHD : l'image (256x256, niveaux de gris) est découpée en 4x4 sous-images.
    Chaque sous-image est découpée en blocs de 8x8 pixels, eux-mêmes vus comme
    2x2 petits blocs de luminance moyenne. Cinq filtres détectent le type de
    contour du bloc : vertical, horizontal, 45°, 135°, non directionnel.
    Un bloc est un contour si la réponse maximale dépasse un seuil (11).

    Vecteur : 16 sous-images x 5 types = 80 valeurs locales (fréquences dans
    [0, 1]) + 5 valeurs globales (moyenne sur l'image, pondérées x5 comme dans
    la distance normative étendue) = 85 valeurs. Distance : L1.
    """

    SIZE = 256
    BLOCK = 8
    GRID = 4
    THRESHOLD = 11.0
    GLOBAL_WEIGHT = 5.0

    @property
    def name(self) -> str:
        return "ehd_4x4_e5"

    def extract(self, image: Image.Image) -> np.ndarray:
        g = np.array(image.convert("L").resize((self.SIZE, self.SIZE), Image.LANCZOS),
                     dtype=np.float32)
        nb = self.SIZE // self.BLOCK          # 32 blocs par côté
        half = self.BLOCK // 2                # 4 pixels par petit bloc

        # Luminance moyenne des 2x2 petits blocs de chaque bloc
        m = g.reshape(nb, 2, half, nb, 2, half).mean(axis=(2, 5))  # (by, sy, bx, sx)
        a, b = m[:, 0, :, 0], m[:, 0, :, 1]
        c, d = m[:, 1, :, 0], m[:, 1, :, 1]

        r2 = np.float32(np.sqrt(2.0))
        responses = np.abs(np.stack([
            a - b + c - d,            # vertical
            a + b - c - d,            # horizontal
            r2 * (a - d),             # 45°
            r2 * (b - c),             # 135°
            2.0 * (a - b - c + d),    # non directionnel
        ]))                           # (5, nb, nb)

        edge_type = responses.argmax(axis=0)
        is_edge = responses.max(axis=0) > self.THRESHOLD

        per = nb // self.GRID             # blocs par sous-image et par côté
        local = np.zeros((self.GRID, self.GRID, 5), dtype=np.float32)
        for t in range(5):
            mask = (is_edge & (edge_type == t)).reshape(self.GRID, per, self.GRID, per)
            local[:, :, t] = mask.sum(axis=(1, 3)) / float(per * per)

        glob = local.reshape(-1, 5).mean(axis=0) * self.GLOBAL_WEIGHT
        return np.concatenate([local.ravel(), glob]).astype(np.float32)


# ── 3. Scalable Color Descriptor ──────────────────────────────────────────────

class ScalableColorDescriptor(Descriptor):
    """
    SCD : histogramme HSV 16x4x4 = 256 bins (normalisé L1), puis transformée de
    Haar le long de l'axe de la teinte (somme/différence de bins voisins,
    récursivement). Les coefficients obtenus sont conservés tous les 256
    (la norme permet de n'en garder que 16, 32, 64, 128 ou 256). Distance : L1.
    """

    BINS = (16, 4, 4)

    @property
    def name(self) -> str:
        return "scd_16_4_4"

    @staticmethod
    def _haar_along_hue(h: np.ndarray) -> np.ndarray:
        """h : (16, 16) = (teinte, S*V). Haar sur l'axe 0 -> même forme."""
        details = []
        cur = h
        while cur.shape[0] > 1:
            even, odd = cur[0::2], cur[1::2]
            details.append(even - odd)
            cur = even + odd
        return np.concatenate([cur] + details[::-1], axis=0)

    def extract(self, image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(self.BINS),
                            [0, 180, 0, 256, 0, 256]).astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        coeffs = self._haar_along_hue(hist.reshape(self.BINS[0], -1))
        return coeffs.ravel().astype(np.float32)