"""API FastAPI du moteur de recherche.

Lancement, depuis la racine du projet :
    uvicorn api.main:app --reload
Documentation interactive : http://localhost:8000/docs
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elasticsearch import Elasticsearch  # noqa: E402
from elasticsearch.exceptions import ConnectionError as ESConnectionError  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from api.schemas import Hit, SearchRequest, SearchResponse  # noqa: E402
import cbir.search as search_module  # noqa: E402
from cbir.search import visual_search  # noqa: E402
from cbir.text_search import ES_URL, search_text  # noqa: E402

# Datasets disponibles : nom -> dossier des images
DATASETS = {"objects": ROOT / "data" / "objects"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Moteur de recherche d'images")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def image_url(dataset, image_id):
    return f"/images/{dataset}/{image_id}"


def category_of(image_id):
    return image_id.split("/")[0] if "/" in image_id else ""


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/web/")


@app.get("/health")
def health():
    try:
        es_ok = Elasticsearch(ES_URL).ping()
    except Exception:
        es_ok = False
    return {"status": "ok", "elasticsearch": bool(es_ok), "datasets": list(DATASETS)}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if req.dataset not in DATASETS:
        raise HTTPException(404, f"Dataset inconnu : {req.dataset}")
    if not req.text and not req.query_image_id:
        raise HTTPException(422, "Donnez des mots-clés et/ou une image requête.")
    if req.text and req.query_image_id:
        raise HTTPException(501, "La fusion texte + image n'est pas encore implémentée.")

    if req.text:  # recherche textuelle seule
        try:
            found = search_text(req.text, dataset=req.dataset, k=req.k)
        except ESConnectionError:
            raise HTTPException(503, "Elasticsearch ne répond pas. Est-il démarré ?")
        hits = [
            Hit(id=h["id"], url=image_url(req.dataset, h["id"]), category=h["category"], score=h["score"])
            for h in found
        ]
    else:  # recherche visuelle seule
        try:
            found = visual_search(req.dataset, req.query_image_id, req.descriptors, k=req.k)
        except NotImplementedError:
            raise HTTPException(501, "La recherche visuelle n'est pas encore disponible.")
        hits = [
            Hit(
                id=h["id"],
                url=image_url(req.dataset, h["id"]),
                category=category_of(h["id"]),
                distance=h["distance"],
            )
            for h in found
        ]
    return SearchResponse(count=len(hits), results=hits)


@app.post("/search/upload", response_model=SearchResponse)
async def search_upload(
    file: UploadFile = File(...),
    dataset: str = Form("objects"),
    descriptors: str = Form("color"),
    k: int = Form(20),
    text: str = Form(""),
    operator: str = Form("AND"),
):
    """Recherche par une image envoyée depuis l'ordinateur."""
    if dataset not in DATASETS:
        raise HTTPException(404, f"Dataset inconnu : {dataset}")
    if text.strip():
        raise HTTPException(501, "La fusion texte + image n'est pas encore implémentée.")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Le fichier envoyé n'est pas une image.")
    data = await file.read()
    if not data:
        raise HTTPException(422, "Fichier vide.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image trop lourde (10 Mo maximum).")

    # Contrat avec la personne A : visual_search_image(dataset, image_bytes, descriptors, k) -> [{"id", "distance"}]
    fn = getattr(search_module, "visual_search_image", None)
    if fn is None:
        raise HTTPException(501, "La recherche par image envoyée n'est pas encore disponible.")
    names = [d.strip() for d in descriptors.split(",") if d.strip()]
    try:
        found = fn(dataset, data, names, k=max(1, min(k, 200)))
    except NotImplementedError:
        raise HTTPException(501, "La recherche par image envoyée n'est pas encore disponible.")
    hits = [
        Hit(id=h["id"], url=image_url(dataset, h["id"]), category=category_of(h["id"]), distance=h["distance"])
        for h in found
    ]
    return SearchResponse(count=len(hits), results=hits)


@app.get("/images/{dataset}/{image_id:path}")
def get_image(dataset: str, image_id: str):
    root = DATASETS.get(dataset)
    if root is None:
        raise HTTPException(404, "Dataset inconnu")
    path = (root / image_id).resolve()
    # sécurité : on refuse de sortir du dossier du dataset (ex. ../../fichier)
    if root.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "Image introuvable")
    return FileResponse(path)


# Interface web (dossier web/), servie par la même application : http://localhost:8000/web/
WEB_DIR = ROOT / "web"
if WEB_DIR.is_dir():
    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")