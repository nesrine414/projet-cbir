import sys
sys.path.insert(0, 'src')
from cbir.search import visual_search, visual_search_image, search

print("=== Test 1 : search() direct ===")
results = search("data/objects/accordion/image_0001.jpg", "hsv", "chi2", k=5)
for path, dist in results:
    print(f"  {dist:.4f}  {path}")

print()
print("=== Test 2 : visual_search() un descripteur ===")
results = visual_search("objects", "accordion/image_0001.jpg", ["color"], k=5)
for r in results:
    print(f"  {r['distance']:.4f}  {r['id']}")

print()
print("=== Test 3 : visual_search() multi-descripteurs ===")
results = visual_search("objects", "accordion/image_0001.jpg", ["color", "texture"], k=5)
for r in results:
    print(f"  rank={r['distance']:.0f}  {r['id']}")

print()
print("=== Test 4 : visual_search_image() (upload) ===")
with open("data/objects/accordion/image_0001.jpg", "rb") as f:
    img_bytes = f.read()
results = visual_search_image("objects", img_bytes, ["color"], k=5)
for r in results:
    print(f"  {r['distance']:.4f}  {r['id']}")

print()
print("Tous les tests OK !")