from abc import ABC, abstractmethod
import numpy as np
from PIL import Image


class Descriptor(ABC):
    """
    Classe abstraite que tous les descripteurs doivent hériter.
    Garantit une interface uniforme pour build_index et search.
    """

    @abstractmethod
    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Extrait le vecteur de signature d'une image.

        Args:
            image: Image PIL (RGB).

        Returns:
            Vecteur numpy 1D de type float32, normalisé.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Identifiant unique du descripteur (ex: 'hsv_8_4_4').
        Utilisé comme clé dans l'index.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
