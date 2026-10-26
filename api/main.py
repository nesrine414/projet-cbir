"""API FastAPI du moteur de recherche.

Lancement, depuis la racine du projet :
    uvicorn api.main:app --reload
Documentation interactive : http://localhost:8000/docs
"""
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elasticsearch import Elasticsearch  # noqa: E402
from elasticsearch.exceptions import ConnectionError as ESConnectionError  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from api.schemas import Hit, SearchRequest, SearchResponse  # noqa: E402
from cbir.data import list_images  # noqa: E402
from cbir.fusion import fuse_results  # noqa: E402
import cbir.search as search_module  # noqa: E402
from cbir.search import (  # noqa: E402
    DescriptorUnavailable,
    IndexUnavailable,
    get_index_status,
    visual_search,
)
from cbir.text_search import ES_URL, search_text  # noqa: E402

# Datasets disponibles : nom -> dossier des images
DATASETS = {
    "objects": ROOT / "data" / "objects",
    "mirflickr": ROOT / "data" / "mirflickr25k",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Moteur de recherche d'images")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def image_url(dataset, image_id):
    return f"/images/{dataset}/{image_id}"


def category_of(image_id):
    first = image_id.split("/", 1)[0] if "/" in image_id else ""
    return "" if first == "mirflickr" else first


@lru_cache(maxsize=4)
def _dataset_catalog(dataset):
    root = DATASETS[dataset]
    if not root.is_dir():
        return ()
    _, ids, categories = list_images(root)
    return tuple(
        {"id": image_id, "url": image_url(dataset, image_id), "category": category}
        for image_id, category in zip(ids, categories)
    )


def _search_visual(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except DescriptorUnavailable as exc:
        raise HTTPException(422, str(exc)) from exc
    except IndexUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(415, "The selected file is not a readable image.") from exc


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


@app.get("/datasets")
def datasets():
    """Describe available image collections and the descriptors indexed for each."""
    result = []
    for name, root in DATASETS.items():
        if not root.is_dir():
            continue
        status = get_index_status(name)
        result.append({
            "id": name,
            "label": "Objects" if name == "objects" else "MIRFLICKR-25K",
            "image_count": len(_dataset_catalog(name)),
            **status,
        })
    return {"datasets": result}


@app.get("/catalog")
def catalog(
    dataset: str = "objects",
    q: str = "",
    offset: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=60),
):
    """Browse stored images so a user can start a search without uploading a file."""
    if dataset not in DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset}")
    items = _dataset_catalog(dataset)
    if not items:
        raise HTTPException(404, f"No images found in dataset: {dataset}")
    query = q.strip().casefold()
    if query:
        items = tuple(
            item for item in items
            if query in item["id"].casefold() or query in item["category"].casefold()
        )
    return {"count": len(items), "offset": offset, "limit": limit, "items": list(items[offset:offset + limit])}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if req.dataset not in DATASETS:
        raise HTTPException(404, f"Dataset inconnu : {req.dataset}")
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
                vis_hits = _search_visual(
                    visual_search,
                    req.dataset, req.query_image_id, req.descriptors, k=req.k, candidate_ids=candidate_ids
                )
            except NotImplementedError:
                raise HTTPException(501, "La recherche visuelle n'est pas encore disponible.")

            hits = [
                Hit(
                    id=h["id"],
                    url=image_url(req.dataset, h["id"]),
                    category=category_of(h["id"]),
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
                vis_hits = _search_visual(
                    visual_search, req.dataset, req.query_image_id, req.descriptors, k=pool_size
                )
            except NotImplementedError:
                raise HTTPException(501, "La recherche visuelle n'est pas encore disponible.")

            fused = fuse_results(text_hits, vis_hits, operator="OR", k=req.k)
            hits = [
                Hit(
                    id=h["id"],
                    url=image_url(req.dataset, h["id"]),
                    category=h["category"] or category_of(h["id"]),
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
                url=h.get("url") or image_url(req.dataset, h["id"]),
                category=h["category"],
                score=h["score"],
            )
            for h in found
        ]
        return SearchResponse(count=len(hits), results=hits)

    # 3. Recherche visuelle seule
    try:
        found = _search_visual(
            visual_search, req.dataset, req.query_image_id, req.descriptors, k=req.k
        )
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
    descriptors: str = Form("auto"),
    k: int = Form(20),
    text: str = Form(""),
    operator: str = Form("AND"),
):
    """Recherche par une image envoyée depuis l'ordinateur."""
    if dataset not in DATASETS:
        raise HTTPException(404, f"Dataset inconnu : {dataset}")
    if not 1 <= k <= 200:
        raise HTTPException(422, "k must be between 1 and 200")
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
                vis_hits = _search_visual(fn, dataset, data, names, k=k, candidate_ids=candidate_ids)
            except NotImplementedError:
                raise HTTPException(501, "La recherche par image envoyée n'est pas encore disponible.")
            hits = [
                Hit(
                    id=h["id"],
                    url=image_url(dataset, h["id"]),
                    category=category_of(h["id"]),
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
                vis_hits = _search_visual(fn, dataset, data, names, k=pool_size)
            except NotImplementedError:
                raise HTTPException(501, "La recherche par image envoyée n'est pas encore disponible.")
            fused = fuse_results(text_hits, vis_hits, operator="OR", k=k)
            hits = [
                Hit(
                    id=h["id"],
                    url=image_url(dataset, h["id"]),
                    category=h["category"] or category_of(h["id"]),
                    score=h.get("score"),
                    distance=h.get("distance"),
                )
                for h in fused
            ]
            return SearchResponse(count=len(hits), results=hits)

    # Cas 2 : Recherche par image seule
    try:
        found = _search_visual(fn, dataset, data, names, k=max(1, min(k, 200)))
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
    path = (root / image_id.replace("\\", "/")).resolve()
    # sécurité : on refuse de sortir du dossier du dataset (ex. ../../fichier)
    if root.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "Image introuvable")
    return FileResponse(path)


# Interface web (dossier web/), servie par la même application : http://localhost:8000/web/
WEB_DIR = ROOT / "web"
if WEB_DIR.is_dir():
    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")
