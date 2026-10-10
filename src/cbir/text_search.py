"""Recherche par mots-clés avec Elasticsearch (7.x).

Test rapide depuis la racine du projet :
    python src/cbir/text_search.py mot_cle
"""
import os
import sys

from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX = os.getenv("ES_INDEX", "cbir_images")

# Analyseur : minuscules, accents ignorés (écrevisse = ecrevisse), pluriels ignorés (dogs = dog).
SETTINGS = {
    "analysis": {
        "analyzer": {
            "folded": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "asciifolding", "porter_stem"],
            }
        }
    }
}

# Un document par image. image_id = chemin relatif de l'image = id dans ids.npy.
MAPPING = {
    "properties": {
        "image_id": {"type": "keyword"},
        "dataset": {"type": "keyword"},
        "category": {"type": "keyword"},  # nom exact du dossier, ex. "car_side"
        "category_words": {"type": "text", "analyzer": "folded"},  # "car side" : permet de chercher "car" ou "ceiling fan"
        "title": {"type": "text", "analyzer": "folded"},
        "tags": {"type": "text", "analyzer": "folded"},
    }
}


def search_text(query, dataset=None, k=20, es=None):
    """Retourne [{"id", "score", "category"}] triés par pertinence."""
    es = es or Elasticsearch(ES_URL)
    must = [{
        "multi_match": {
            "query": query,
            "fields": ["category_words^3", "tags^2", "title"],
            "fuzziness": "AUTO:4,8",
        }
    }]
    flt = [{"term": {"dataset": dataset}}] if dataset else []
    body = {"query": {"bool": {"must": must, "filter": flt}}, "size": k}
    res = es.search(index=INDEX, body=body)
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