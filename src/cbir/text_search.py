"""Recherche par mots-clés avec Elasticsearch (7.x).

Test rapide depuis la racine du projet :
    python src/cbir/text_search.py mot_cle
"""
import os
import sys

from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX = os.getenv("ES_INDEX", "cbir_images")

# Un document par image. image_id = chemin relatif de l'image = id dans ids.npy.
MAPPING = {
    "properties": {
        "image_id": {"type": "keyword"},
        "dataset": {"type": "keyword"},
        "category": {"type": "keyword"},  # nom exact du dossier, ex. "car_side"
        "category_words": {"type": "text"},  # "car side" : permet de chercher "car" ou "ceiling fan"
        "title": {"type": "text"},
        "tags": {"type": "text"},
    }
}


def search_text(query, dataset=None, k=20, es=None):
    """Retourne [{"id", "score", "category"}] triés par pertinence."""
    es = es or Elasticsearch(ES_URL)
    must = [{
        "multi_match": {
            "query": query,
            "fields": ["category_words^3", "tags^2", "title"],
            "fuzziness": "AUTO",
        }
    }]
    flt = [{"term": {"dataset": dataset}}] if dataset else []
    query_body = {"bool": {"must": must, "filter": flt}}
    res = es.search(index=INDEX, query=query_body, size=k)
    return [
        {
            "id": h["_source"]["image_id"],
            "score": h["_score"],
            "category": h["_source"].get("category", ""),
        }
        for h in res["hits"]["hits"]
    ]


if __name__ == "__main__":
    texte = " ".join(sys.argv[1:]) or "test"
    resultats = search_text(texte, k=10)
    print(len(resultats), "résultat(s) pour", repr(texte))
    for r in resultats:
        print(r)