"""Index dataset images and per-image text metadata into Elasticsearch."""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elasticsearch import Elasticsearch, helpers  # noqa: E402
from cbir.data import list_images  # noqa: E402
from cbir.text_search import ES_URL, INDEX, MAPPING, SETTINGS  # noqa: E402


def load_tags():
    path = ROOT / "config" / "category_tags.json"
    if not path.exists():
        print("config/category_tags.json not found; continuing without extra category tags")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def objects_documents(root, dataset, tags):
    files, ids, categories = list_images(root)
    for image_file, image_id, category in zip(files, ids, categories):
        yield {
            "_index": INDEX,
            "_id": f"{dataset}:{image_id}",
            "_source": {
                "image_id": image_id,
                "image_url": f"/images/{dataset}/{image_id}",
                "dataset": dataset,
                "category": category,
                "category_words": category.replace("_", " ").replace("-", " "),
                "title": image_file.stem.replace("_", " ").replace("-", " "),
                "tags": tags.get(category, ""),
            },
        }


def mirflickr_documents(root, dataset="mirflickr"):
    """Index per-image Flickr tags and manual MIRFLICKR topic labels."""
    root = Path(root)
    labels_by_image = defaultdict(set)
    for annotation_file in sorted((root / "annotations").glob("*.txt")):
        if annotation_file.stem.lower() == "readme":
            continue
        label = annotation_file.stem.removesuffix("_r1").replace("_", " ")
        for line in annotation_file.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                image_number = int(line.strip())
            except ValueError:
                continue
            labels_by_image[f"mirflickr/im{image_number}.jpg"].add(label)

    tag_root = root / "mirflickr" / "meta" / "tags"
    files, ids, _ = list_images(root)
    for image_file, image_id in zip(files, ids):
        match = re.fullmatch(r"im(\d+)", Path(image_id).stem)
        number = match.group(1) if match else ""
        tag_file = tag_root / f"tags{number}.txt"
        try:
            user_tags = [
                line.strip()
                for line in tag_file.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
        except FileNotFoundError:
            user_tags = []

        labels = sorted(labels_by_image.get(image_id, ()))
        yield {
            "_index": INDEX,
            "_id": f"{dataset}:{image_id}",
            "_source": {
                "image_id": image_id,
                "image_url": f"/images/{dataset}/{image_id}",
                "dataset": dataset,
                "category": " ".join(labels),
                "category_words": " ".join(labels),
                "title": image_file.stem,
                "tags": " ".join([*user_tags, *labels]),
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--dataset", choices=("objects", "mirflickr"), default="objects")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    default_root = ROOT / "data" / ("mirflickr25k" if args.dataset == "mirflickr" else "objects")
    root = Path(args.root) if args.root else default_root

    es = Elasticsearch(ES_URL)
    if args.reset and es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)
        print("Deleted index:", INDEX)
    if not es.indices.exists(index=INDEX):
        es.indices.create(index=INDEX, body={"settings": SETTINGS, "mappings": MAPPING})
        print("Created index:", INDEX)

    if args.dataset == "mirflickr":
        annotation_files = list((root / "annotations").glob("*.txt"))
        tag_dir = root / "mirflickr" / "meta" / "tags"
        if not annotation_files or not tag_dir.is_dir():
            raise SystemExit(f"MIRFLICKR annotations or per-image tags were not found under {root}")
        actions = mirflickr_documents(root, args.dataset)
    else:
        tags = load_tags()
        categories = {category for category in list_images(root)[2] if category}
        missing = sorted(categories - set(tags))
        if missing:
            print("Categories without extra keywords:", ", ".join(missing))
        actions = objects_documents(root, args.dataset, tags)

    count, _ = helpers.bulk(es, actions)
    es.indices.refresh(index=INDEX)
    print(count, "documents indexed for dataset", args.dataset)


if __name__ == "__main__":
    main()