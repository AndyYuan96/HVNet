import numpy as np
import time
a = np.random.rand(3,3)
b = np.random.rand(100000,3)
t1 = time.time()
np.dot(b,a)
t2 = time.time()
print(t2 - t1)
