import numpy as np


test = np.arange(1,20,2)[:,np.newaxis]
test[-1] += 10

print(test)

test_diff = np.diff(test, axis = 0, append=2)

print(test_diff)