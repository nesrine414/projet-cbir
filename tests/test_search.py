import sys; sys.path.insert(0, 'src')
from cbir.search import search

results = search(
    "data/objects/accordion/image_0001.jpg",
    descriptor_name="hsv",
    distance_name="chi2",
    k=5
)

for path, dist in results:
    print(f"{dist:.4f}  {path}")
    