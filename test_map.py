"""
Testing for interruptable multiprocessing 
"""

import time
import multiprocessing as mltproc  
import numpy as np 


def square(x):
    try:
        time.sleep(1)
        return x*x
    except KeyboardInterrupt:
        return None

test_x = np.arange(30)
print(test_x)
t0 = time.time()
with mltproc.Pool(3) as pool:
    # out = []
    try:
        # for result in pool.imap(square, test_x):
        #     out.append(result)    
        out = pool.map(square, test_x)
    except KeyboardInterrupt:
        print("end")
print(time.time() - t0)
print(out)
