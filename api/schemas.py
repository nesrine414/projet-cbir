"""Modèles JSON de requête et de réponse de l'API (le contrat avec l'interface web)."""
from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    dataset: str = "objects"
    text: Optional[str] = Field(None, description="Mots-clés (recherche textuelle)")
    query_image_id: Optional[str] = Field(None, description="Id d'une image de la base (recherche visuelle)")
    descriptors: List[str] = Field(default_factory=lambda: ["color"])
    operator: str = Field("AND", description="AND ou OR, pour combiner texte et image")
    k: int = Field(20, ge=1, le=200)


class Hit(BaseModel):
    id: str
    url: str
    category: str = ""
    score: Optional[float] = None  # recherche textuelle : plus grand = plus pertinent
    distance: Optional[float] = None  # recherche visuelle : plus petit = plus proche


class SearchResponse(BaseModel):
    count: int
    results: List[Hit]