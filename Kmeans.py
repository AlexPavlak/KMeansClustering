import numpy as np
import matplotlib.pyplot as plt

#X = data, k = number of clusters, c = clusters
def mykmeans(X,k,c):
    #add a column for each points cluster to the data 
    pointsWithClusters = np.hstack((X,np.zeros((len(X),1))))


    #for every point in X
    for i in X:
        
        #calculate the distance to every center in C
        for j in c:
            if(j==0):
                pointsWithClusters[i][2] = np.linalg.norm(X[i]-c[j])
            elif(pointsWithClusters[i][2] > np.linalg.norm(X[i]-c[j])):
                pointsWithClusters[i][2] = np.linalg.norm(X[i]-c[j])

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


 
