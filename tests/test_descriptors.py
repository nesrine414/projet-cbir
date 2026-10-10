#tester le descripteur d'image HSV
from PIL import Image
from cbir.descriptors.color import HSVDescriptor

d = HSVDescriptor()
v = d.extract(Image.open("data/objects/butterfly/image_0010.jpg"))
print(d, v.shape, v.sum())   # attendu : (128,) et une somme ≈ 1.0
import sys
sys.path.insert(0, 'src')


#tester le descripteur de texture LBP
from PIL import Image
from cbir.descriptors.texture import LBPDescriptor
import numpy as np
img = Image.open(r'data/objects/accordion/image_0001.jpg')
desc = LBPDescriptor()
print(desc)  # doit afficher : LBPDescriptor(name='lbp_P24_R3')
vec = desc.extract(img)
print("Shape :", vec.shape)    # doit être (26,)
print("Somme :", vec.sum())    # doit être 1.0
print("Vecteur :", vec)

img2 = Image.open(r'data/objects/accordion/image_0002.jpg')
vec2 = desc.extract(img2)
print("Vecteurs différents ?", not np.allclose(vec, vec2))  # doit être True

#tester le descripteur de forme HOG
import sys
sys.path.insert(0, 'src')
from PIL import Image
from cbir.descriptors.shape import HOGDescriptor
import numpy as np

img = Image.open(r'data/objects/accordion/image_0001.jpg')
desc = HOGDescriptor()
vec = desc.extract(img)
print("Shape :", vec.shape)   # (1764,)
print("Somme :", vec.sum())   # pas 1.0 ici, normalisé autrement

