import func
import de_func
import numpy as np

n = 10
d = 2
f = 0.9
c = 0.9
iter = 1000
g = 100

de = de_func.DE(n, d, f, c, func.sphere, iter, g)
de.get_graph()