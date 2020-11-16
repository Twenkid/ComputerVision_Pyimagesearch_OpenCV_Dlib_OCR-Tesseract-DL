#graph test, Todor, 3-6-2017
import numpy as np
import scipy.sparse.csgraph

#x = np.array([1, 2, 3, -1, 5, 6])

x = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0,0,0,1], [0, 0,0, 0]], np.int32)
#x = np.array([[0, 1, 1], [1, 1, 1], [1,1,1]], np.int32)


#scipy.sparse.csgraph.dijkstra(csgraph, directed=True, indices=None, return_predecessors=False, unweighted=False)

path, pred = scipy.sparse.csgraph.dijkstra(x, directed=True, indices=None, return_predecessors=True, unweighted=False)

print("Path", path)
print("Pred", pred)

