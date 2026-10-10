import sys; sys.path.insert(0, 'src')
import numpy as np
from cbir.distances import chi2, l1, l2

a = np.array([0.5, 0.3, 0.2])
b = np.array([0.5, 0.3, 0.2])

print(chi2(a, b))   # 0.0  ← même vecteur
print(l1(a, b))     # 0.0
print(l2(a, b))     # 0.0

c = np.array([0.1, 0.6, 0.3])
print(chi2(a, c))   # > 0
print(l1(a, c))     # > 0
print(l2(a, c))     # > 0