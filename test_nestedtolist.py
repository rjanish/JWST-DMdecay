#!/usr/bin/env python3

import numpy as np 

from nestedtolist import nestedtolist

if __name__ == "__main__":
	trials = []
	trials += [5]
	trials += [np.arange(5)]
	trials += [["test", [3, 4], np.array([[1, 1], [2, 3]])]]
	trials += [["test", [3, np.array([4])], np.array([[1, 1], [2, 3]])]]
	for c, t in enumerate(trials):
		print(F"trial {c}:")
		print(t)
		print(type(t)) 
		tl = nestedtolist(t)
		print(tl)
		print(type(tl))
		print()