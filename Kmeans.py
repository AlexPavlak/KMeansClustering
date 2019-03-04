import numpy as np
import matplotlib.pyplot as plt

#X = data, k = number of clusters, c = clusters
def mykmeans(X,k,c):
    #add a column for each points cluster to the data 
    pointsWithClusters = np.hstack((X,np.zeros((len(X),1))))

    #create an array to store the sums of points in a cluster
    #as well as the number of points in the cluster
    centerSets = np.zeros((k,1+np.size(X,1)))

    rowCount=0
    #These 2 loops perform step #2 from the K-means algorithm
    #For each point in X, it will calculate and store the closest
    #center
    for i in X:
        
        #keep track of what center we are currently working with for each
        #row of X
        centerCount = 0
        #calculate the distance from each point to every center in C
        
        for j in c:
            #calculate the distance to the first center every and assign it
            #as the closest center
            if(centerCount==0):
                distanceToClosestCenter = np.linalg.norm(i-j)
                assignedCenter = 0
                centerCount += 1
            else:
                #For every center after the first, calculate the distance
                #and compare it to the current closest distance to center.
                #save which ever distance is smaller as the cloesest distance to center
                distToCenter = np.linalg.norm(i-j)
                if(distToCenter < distanceToClosestCenter):
                    pointsWithClusters[rowCount][2] = centerCount
                    distanceToClosestCenter = distToCenter
                    assignedCenter = centerCount
                centerCount+=1
        #add the newly assigned points values to the set of points for that cluster
        centerSets[assignedCenter][0] += 1
        centerSets[assignedCenter,1:] += i
        rowCount += 1
    np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
    print(pointsWithClusters)
    print(centerSets)
    #perform the update step

###MAIN###
#Given parameters for generating 1st set of gausian numbers
mean1 = np.array([1,0])
dev1 = np.array([[0.9,0.4],[0.4,0.9]])

#Given parameters for generating 2nd set of gausian numbers
mean2 = np.array([0,1.5])
dev2 = ([[0.9,0.4],[0.4,0.9]])

set1 = np.random.multivariate_normal(mean1,dev1,500)
set2 = np.random.multivariate_normal(mean2,dev2,500)

#plt.scatter(set1[:,0],set1[:,1])
#plt.show()

mykmeans(set1,2,[[10,10],[-10,-10]])


 
