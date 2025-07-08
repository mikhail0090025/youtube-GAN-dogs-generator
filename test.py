from scipy.stats import entropy
import numpy as np

print(entropy(np.ones(100), np.random.random(100)))
print(entropy(np.random.random(100), np.random.random(100)))
print(entropy(np.random.standard_normal((100, 100)), np.random.random((100, 100))))
print(entropy(np.random.standard_normal(100), np.random.standard_normal(100)))