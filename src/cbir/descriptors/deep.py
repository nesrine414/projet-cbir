import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from .base import Descriptor


class ResNetDescriptor(Descriptor):
    """
    Descripteur visuel sémantique basé sur ResNet18 (Deep Learning).

    Vecteur : 512 valeurs normalisées L2.
    Distance : Similarité Cosinus d = 0.5 * (1 - dot(q, db)).
    Capture la sémantique de l'objet (rhino, fauteuil, moto, tournesol)
    de façon invariante aux variations de fond, d'angle et d'éclairage.
    """

    def __init__(self):
        self._model = None
        self._preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _ensure_model(self):
        if self._model is None:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            base.eval()
            self._model = torch.nn.Sequential(*(list(base.children())[:-1]))

    @property
    def name(self) -> str:
        return "resnet18_512d"

    def extract(self, image: Image.Image) -> np.ndarray:
        self._ensure_model()
        img_rgb = image.convert("RGB")
        tensor = self._preprocess(img_rgb).unsqueeze(0)
        with torch.no_grad():
            emb = self._model(tensor).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        return emb.astype(np.float32)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        self._ensure_model()
        tensors = torch.stack([self._preprocess(img.convert("RGB")) for img in images])
        with torch.no_grad():
            embs = self._model(tensors).squeeze().cpu().numpy()
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        np.maximum(norms, 1e-10, out=norms)
        embs = embs / norms
        return embs.astype(np.float32)
