import numpy as np
import matplotlib.pyplot as plt

#Given parameters for generating 1st set of gausian numbers
mean1 = np.array([1,0])
dev1 = np.array(([0.9,0.4],[0.4,0.9]))

#Given parameters for generating 2nd set of gausian numbers
mean2 = np.array([0,1.5])
dev2 = (([0.9,0.4],[0.4,0.9]))

set1 = np.random.multivariate_normal(mean1,dev1,500)
set2 = np.random.multivariate_normal(mean2,dev2,500)

)