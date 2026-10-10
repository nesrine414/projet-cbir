"""Indexe les images d'un dataset dans Elasticsearch (phase hors ligne, côté texte).

Usage, depuis la racine du projet :
    python scripts/ingest_es.py --root data/objects --dataset objects
Options : --reset pour supprimer l'index avant de le recréer.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elasticsearch import Elasticsearch, helpers  # noqa: E402

from cbir.data import list_images  # noqa: E402
from cbir.text_search import ES_URL, INDEX, MAPPING  # noqa: E402


def documents(root, dataset):
    files, ids, cats = list_images(root)
    for f, image_id, cat in zip(files, ids, cats):
        yield {
            "_index": INDEX,
            "_id": f"{dataset}:{image_id}",  # relancer le script écrase au lieu de dupliquer
            "_source": {
                "image_id": image_id,
                "dataset": dataset,
                "category": cat,
                "category_words": cat.replace("_", " ").replace("-", " "),
                "title": f.stem.replace("_", " ").replace("-", " "),
                "tags": "",
            },
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/objects")
    ap.add_argument("--dataset", default="objects")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()

    es = Elasticsearch(ES_URL)
    if a.reset and es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)
        print("index supprimé")
    if not es.indices.exists(index=INDEX):
        es.indices.create(index=INDEX, mappings=MAPPING)
        print("index créé :", INDEX)

    n, _ = helpers.bulk(es, documents(a.root, a.dataset))
    es.indices.refresh(index=INDEX)
    print(n, "documents indexés pour le dataset", a.dataset)


if __name__ == "__main__":
    main()