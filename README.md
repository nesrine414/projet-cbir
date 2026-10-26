# Framefind — Content Based Image Retrieval

Framefind is a client/server image search app built around the requirements in pages 56–61 of the supplied multimedia indexing course. It searches Objects and MIRFLICKR-25K by text, by image, or with both query types. The web client lets you pick an image from the collection or upload one, choose visual descriptors, and click a result to run another search.

## What it does

- **Image retrieval:** HSV colour, LBP texture, HOG shape, and MPEG-7 Color Layout, Edge Histogram, and Scalable Color descriptors.
- **Text retrieval:** Elasticsearch matches category names, titles, and configured tags. Text metadata is available for Objects through `config/category_tags.json`.
- **Combined retrieval:** AND filters text matches before visual reranking. OR merges text and image rankings with reciprocal rank fusion.
- **Image catalog:** Browse and filter stored images before running a visual search.
- **Search acceleration:** A VP-tree is built lazily for single-descriptor searches on collections up to 12,000 images. Multi-descriptor and larger searches use aligned NumPy matrices and partial top-k selection.
- **API and web client:** FastAPI serves JSON endpoints, images, and the static browser app.

## Project layout

```text
api/                 FastAPI routes and request/response models
config/              Search settings and Objects category tags
data/objects/        Objects image collection
data/mirflickr25k/   MIRFLICKR-25K images, tags, and annotations
index/               Generated descriptor indexes (not committed)
scripts/             Index building, Elasticsearch ingest, evaluation
src/cbir/            Descriptors, ranking, fusion, and index structures
web/                 Browser client
```

## Run locally

Install Python dependencies, then prepare the visual index and optional text index:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Build all classic and MPEG-7 descriptors for Objects.
python scripts/build_index.py --dataset objects

# Optional: add the Objects categories and tags to Elasticsearch.
docker compose up -d elasticsearch
python scripts/ingest_es.py --root data/objects --dataset objects

uvicorn api.main:app --reload
```

Open `http://localhost:8000/web/`. API documentation is at `http://localhost:8000/docs`.

If an Objects index already exists and you only want to add MPEG-7 signatures while keeping the other descriptors, run:

```powershell
python scripts/add_mpeg7_to_index.py objects
```

Build a visual index for the extracted MIRFLICKR-25K images with MPEG-7 descriptors:

```powershell
python scripts/build_index.py --dataset mirflickr --descriptors cld ehd scd
python scripts/ingest_es.py --dataset mirflickr
```

The ingest command adds each image's Flickr tags and manual topic annotations to Elasticsearch. The app uses the 25K collection under `data/mirflickr25k`; the older thumbnail folder under `data/mirflickr` is ignored.

## Docker

Build the visual index on the host first so the generated file is available in `index/`, then run:

```powershell
docker compose up --build
```

The Compose stack runs Elasticsearch and the FastAPI app. The local `data/` and `index/` directories are mounted into the app container.

## API summary

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | API and Elasticsearch status |
| `GET` | `/datasets` | Image counts and indexed descriptor options |
| `GET` | `/catalog?dataset=objects&q=butterfly` | Browse or filter stored images |
| `POST` | `/search` | Text, stored-image, or combined search |
| `POST` | `/search/upload` | Search using an uploaded image, optionally with text |
| `GET` | `/images/{dataset}/{image_id}` | Serve an image from a collection |

## Building indexes

`scripts/build_index.py` supports `objects` and `mirflickr`. Its default descriptor set is HSV, LBP, HOG, and the three MPEG-7 descriptors. ResNet is an optional extra and requires `torch` and `torchvision`.

Use `--append` when rebuilding selected signatures into an existing index. For example, this preserves other Objects descriptors while replacing the three MPEG-7 entries:

```powershell
python scripts/build_index.py --dataset objects --descriptors cld ehd scd --append
```

Generated index files are ignored by Git. Restart the API after rebuilding an index so it reloads the updated signatures.
