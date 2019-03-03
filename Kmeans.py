import numpy as np
import matplotlib.pyplot as plt

#X = data, k = number of clusters, c = clusters
def mykmeans(X,k,c):
    #add a column for each points cluster to the data 
    pointsWithClusters = np.hstack((X,np.zeros((len(X),1))))

    rowCount=0
    #for every point in X
    for i in X:
        
        #keep track of what center we are currently working with for each
        #row of X
        centerCount = 0
        #calculate the distance from each point to every center in C
        
        for j in c:
            
            if(centerCount==0):
                distanceToClosestCenter = np.linalg.norm(i-j)
                centerCount += 1
            else:
                distToCenter = np.linalg.norm(i-j)
                if(distToCenter < distanceToClosestCenter):
                    pointsWithClusters[rowCount][2] = centerCount
                    distanceToClosestCenter = distToCenter
                centerCount+=1
        rowCount += 1
    np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
    print(pointsWithClusters) 

###MAIN###
#Given parameters for generating 1st set of gausian numbers
mean1 = np.array([1,0])
dev1 = np.array([[0.9,0.4],[0.4,0.9]])

#Given parameters for generating 2nd set of gausian numbers
mean2 = np.array([0,1.5])
dev2 = ([[0.9,0.4],[0.4,0.9]])

set1 = np.random.multivariate_normal(mean1,dev1,500)
set2 = np.random.multivariate_normal(mean2,dev2,500)

mykmeans(set1,2,[[10,10],[-10,10]])


 
