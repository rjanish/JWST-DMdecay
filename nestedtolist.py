
import numpy as np 


def nestedtolist(l):
    """
    Convert all ndarrays into lists within an arbitrarily 
    nested list of ndarrays and other objects. 
    This may modify the passed list in place.   
    """
    if isinstance(l, np.ndarray):
        l = l.tolist()
    if isinstance(l, list):
        for i in range(len(l)):
            l[i] = nestedtolist(l[i])
    return l