"""API FastAPI du moteur de recherche.

Lancement, depuis la racine du projet :
    uvicorn api.main:app --reload
Documentation interactive : http://localhost:8000/docs
"""
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elasticsearch import Elasticsearch  # noqa: E402
from elasticsearch.exceptions import ConnectionError as ESConnectionError  # noqa: E402
from elasticsearch.exceptions import ElasticsearchException  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from api.schemas import Hit, SearchRequest, SearchResponse  # noqa: E402
from cbir.fusion import fuse_results  # noqa: E402
from cbir.index.store import IndexNotFoundError  # noqa: E402
import cbir.search as search_module  # noqa: E402
from cbir.search import visual_search  # noqa: E402
from cbir.text_search import ES_URL, INDEX, search_text  # noqa: E402

# Datasets disponibles : nom -> dossier des images
DATASETS = {
    "objects": ROOT / "data" / "objects",
    "mirflickr": ROOT / "data" / "mirflickr",
}
TEXT_SEARCH_DATASETS = {"objects"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
logger = logging.getLogger(__name__)

app = FastAPI(title="Moteur de recherche d'images")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def image_url(dataset, image_id):
    return f"/images/{dataset}/{image_id}"


def category_of(image_id, dataset="objects"):
    if dataset == "mirflickr":
        return ""
    return image_id.split("/")[0] if "/" in image_id else ""


def require_text_search_dataset(dataset):
    if dataset not in TEXT_SEARCH_DATASETS:
        raise HTTPException(
            422,
            f"La recherche textuelle n'est pas disponible pour {dataset} : "
            "ce dataset ne contient pas d'annotations textuelles. "
            "Choisissez Objects pour rechercher par mots-clés.",
        )


def run_visual_search(dataset, search_fn, *args, **kwargs):
    try:
        return search_fn(dataset, *args, **kwargs)
    except IndexNotFoundError as exc:
        raise HTTPException(
            503,
            f"L'index visuel de {dataset} n'est pas encore construit. "
            f"Lancez : python scripts/build_index.py --dataset {dataset}",
        ) from exc


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/web/")


@app.get("/health")
def health():
    es = Elasticsearch(ES_URL, request_timeout=2)
    try:
        es.info()
    except ElasticsearchException as exc:
        logger.warning("Elasticsearch health check failed: %s", type(exc).__name__)
        es.close()
        return {
            "status": "degraded",
            "elasticsearch": False,
            "elasticsearch_error": type(exc).__name__,
            "text_index": False,
            "datasets": list(DATASETS),
        }

    try:
        index_exists = bool(es.indices.exists(index=INDEX))
    except ElasticsearchException as exc:
        logger.warning("Elasticsearch index check failed: %s", type(exc).__name__)
        index_exists = False
        index_error = type(exc).__name__
    else:
        index_error = None
    finally:
        es.close()

    result = {
        "status": "ok" if index_exists else "degraded",
        "elasticsearch": True,
        "text_index": index_exists,
        "datasets": list(DATASETS),
    }
    if index_error:
        result["text_index_error"] = index_error
    return result


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if req.dataset not in DATASETS:
        raise HTTPException(404, f"Dataset inconnu : {req.dataset}")
    if req.text:
        require_text_search_dataset(req.dataset)
    if not req.text and not req.query_image_id:
        raise HTTPException(422, "Donnez des mots-clés et/ou une image requête.")

    # 1. Recherche combinée : Texte ET/OU Image
    if req.text and req.query_image_id:
        if req.operator.upper() == "AND":
            try:
                text_hits = search_text(req.text, dataset=req.dataset, k=200)
            except ESConnectionError:
                raise HTTPException(503, "Elasticsearch ne répond pas. Est-il démarré ?")
            if not text_hits:
                return SearchResponse(count=0, results=[])

            candidate_ids = [h["id"] for h in text_hits]
            try:
                vis_hits = run_visual_search(
                    req.dataset, visual_search, req.query_image_id, req.descriptors,
                    k=req.k, candidate_ids=candidate_ids,
                )
            except NotImplementedError:
                raise HTTPException(501, "La recherche visuelle n'est pas encore disponible.")

            hits = [
                Hit(
                    id=h["id"],
                    url=image_url(req.dataset, h["id"]),
                    category=category_of(h["id"], req.dataset),
                    distance=h["distance"],
                )
                for h in vis_hits
            ]
            return SearchResponse(count=len(hits), results=hits)
        else:  # OR
            pool_size = max(req.k * 3, 60)
            try:
                text_hits = search_text(req.text, dataset=req.dataset, k=pool_size)
            except ESConnectionError:
                raise HTTPException(503, "Elasticsearch ne répond pas. Est-il démarré ?")
            try:
                vis_hits = run_visual_search(
                    req.dataset, visual_search, req.query_image_id,
                    req.descriptors, k=pool_size,
                )
            except NotImplementedError:
                raise HTTPException(501, "La recherche visuelle n'est pas encore disponible.")

            fused = fuse_results(text_hits, vis_hits, operator="OR", k=req.k)
            hits = [
                Hit(
                    id=h["id"],
                    url=image_url(req.dataset, h["id"]),
                    category=category_of(h["id"], req.dataset),
                    score=h.get("score"),
                    distance=h.get("distance"),
                )
                for h in fused
            ]
            return SearchResponse(count=len(hits), results=hits)

    # 2. Recherche textuelle seule
    if req.text:
        try:
            found = search_text(req.text, dataset=req.dataset, k=req.k)
        except ESConnectionError:
            raise HTTPException(503, "Elasticsearch ne répond pas. Est-il démarré ?")
        hits = [
            Hit(
                id=h["id"],
                url=image_url(req.dataset, h["id"]),
                category=category_of(h["id"], req.dataset),
                score=h["score"],
            )
            for h in found
        ]
        return SearchResponse(count=len(hits), results=hits)

    # 3. Recherche visuelle seule
    try:
        found = run_visual_search(
            req.dataset, visual_search, req.query_image_id, req.descriptors, k=req.k
        )
    except NotImplementedError:
        raise HTTPException(501, "La recherche visuelle n'est pas encore disponible.")
    hits = [
        Hit(
            id=h["id"],
            url=image_url(req.dataset, h["id"]),
            category=category_of(h["id"], req.dataset),
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
        require_text_search_dataset(dataset)
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

    # Cas 1 : Recherche combinée Texte ET/OU Image uploadée
    if text.strip():
        if operator.upper() == "AND":
            try:
                text_hits = search_text(text.strip(), dataset=dataset, k=200)
            except ESConnectionError:
                raise HTTPException(503, "Elasticsearch ne répond pas. Est-il démarré ?")
            if not text_hits:
                return SearchResponse(count=0, results=[])

            candidate_ids = [h["id"] for h in text_hits]
            try:
                vis_hits = run_visual_search(
                    dataset, fn, data, names, k=k, candidate_ids=candidate_ids
                )
            except NotImplementedError:
                raise HTTPException(501, "La recherche par image envoyée n'est pas encore disponible.")
            hits = [
                Hit(
                    id=h["id"],
                    url=image_url(dataset, h["id"]),
                    category=category_of(h["id"], dataset),
                    distance=h["distance"],
                )
                for h in vis_hits
            ]
            return SearchResponse(count=len(hits), results=hits)
        else:  # OR
            pool_size = max(k * 3, 60)
            try:
                text_hits = search_text(text.strip(), dataset=dataset, k=pool_size)
            except ESConnectionError:
                raise HTTPException(503, "Elasticsearch ne répond pas. Est-il démarré ?")
            try:
                vis_hits = run_visual_search(dataset, fn, data, names, k=pool_size)
            except NotImplementedError:
                raise HTTPException(501, "La recherche par image envoyée n'est pas encore disponible.")
            fused = fuse_results(text_hits, vis_hits, operator="OR", k=k)
            hits = [
                Hit(
                    id=h["id"],
                    url=image_url(dataset, h["id"]),
                    category=category_of(h["id"], dataset),
                    score=h.get("score"),
                    distance=h.get("distance"),
                )
                for h in fused
            ]
            return SearchResponse(count=len(hits), results=hits)

    # Cas 2 : Recherche par image seule
    try:
        found = run_visual_search(
            dataset, fn, data, names, k=max(1, min(k, 200))
        )
    except NotImplementedError:
        raise HTTPException(501, "La recherche par image envoyée n'est pas encore disponible.")
    hits = [
        Hit(
            id=h["id"],
            url=image_url(dataset, h["id"]),
            category=category_of(h["id"], dataset),
            distance=h["distance"],
        )
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