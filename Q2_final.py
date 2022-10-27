import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import id3a
import cartq2

#[[[1, 2, 4, 5],[5,2,1,7]],[[3,4,1,6]]]
def clusterAvg(x):
    res = []
    for i in range(len(x)):
        res.append(vectorAvg(x[i]))
    return res
    
def normalAvg(x):
    N = len(x)
    s = 0
    for i in range(N):
        s = s + x[i]
    return s/N
    
def vectorAvg(x):
    res = []
    C = len(x[0])
    for i in range(C):
       col = extractColumn(x,i)
       res.append(normalAvg(col))
    return res
    
def extractColumn(T,index):
    R = len(T)
    col = []
    for i in range(R):
        col.append(T[i][index])
    return col
    
def centriod(x,y):#[1,2,3],[4,5,6]
    sx = 0
    N = len(x)
    sy = 0
    for i in range(N):
        sx = sx + x[i]
        sy = sy + y[i]
    return (sx/N,sy/N)

def similarity(x,y):#[1,2,3],[5,6,9]
    s = 0
    for i in range(len(x)):
        s = s + abs(x[i]-y[i])
    return s

def euclideanDistance(x,y):# [1,2,5,3],[2,4,1,6]
    s = 0 
    for i in range(len(x)):
        s = s + (x[i]-y[i])**2
    return np.sqrt(s)
    
def difference(x,y):#[1,2,3],[5,6,9]
    res = []
    for i in range(len(x)):
        res.append(abs(x[i]-y[i]))
    return res


def initCluster(T,k):
    res = []
    for i in range(k):
        res.append(T[i])
    return res
    
def form3DList(n):
    res = []
    for i in range(n):
        row = []
        res.append(row)
    return res

def threshold(x,t):    #[[1,0,4,2],[0,2,1,4]] == > [7,7]
    res = []
    for i in range(len(x)):
        res.append(normalAvg(x[i])*len(x[0]))
    return res 
    
def diffInCentriod(x,y):#[[1,2,4,5],[5,2,1,7]],[[1,2,4,5],[5,2,1,7]]
    res = []
    for i in range(len(x)):
        res.append(difference(x[i],y[i]))
    return res 
    
def addToCluster(newdata,existingdata,clusterId):
    existingdata[clusterId].append(newdata)
    return existingdata

def kMeans(data,centriodCoor,r):# [[1 2 4 5],[5,2,1,7],[3,1,25,1],[4,5,2,1]], centriod k = 2; [[1,2,4,5],[5,2,1,7]]
    k = len(centriodCoor)
    clusters = form3DList(k)
    sampleNumber = len(data)
    for i in range(sampleNumber):
        mn = 99999
        mnIndex = -1
        for j in range(k):#k =2 j =0 ,1
            sim = euclideanDistance(data[i],centriodCoor[j])
            # print("similarity = ",sim)
            if sim < mn:
                mn = sim
                mnIndex = j
            # print("MinIndex = ",mnIndex)
        clusters = addToCluster(data[i],clusters,mnIndex)
    # print("Round = ",r)
    # print(clusters)
    clusterCentriod = clusterAvg(clusters)
    # print("Final Cluster = ",clusterCentriod)
    if r <= 30:
        r = r + 1
        return kMeans(data,clusterCentriod,r) # data: raw data, clusterCentroid, r:iterations (manually set)
    else:
        return [clusterCentriod, clusters]
        # print(clusters)
        # with open("clusters_file.csv","w+") as my_csv:
        #     write = csv.writer(my_csv,delimiter=',')
        #     write.writerows(clusters)
    




    
file = open("covid_updated_file.csv")
T = np.loadtxt(file, delimiter=",")
print(len(T))
# 
#       
# print(form3DList(3)

cent = (initCluster(T,10)) #table data and clusters to create

# # cent = [[27,10,15.66,8.66],[39.25,73,48,19.75],[24.4,73.2,50,76.4]]
# print("Initial Cluster = ",cent)
# print(kMeans(T,cent,0))

cc = kMeans(T, cent, 0)
# print(cc[1][0])
tree = []
for i in range(len(cc[1])):
    a, b = id3a.id3a_main(cc[1][i])
    tree.append(a)

print("RMSE Kmeans with ID3 = ",b)

cc = kMeans(T, cent, 0)
# print(cc[1][0])
tree = []
for i in range(len(cc[1])):
    a, b = CART_AQuest2.CART_main(cc[1][i])
    tree.append(a)

print("RMSE Kmeans with CART = ",b)
# print(cc[1])
# day = 1
# for i in range(len(cc[1])):#for each cluster
#     # clt = extractColumn(cc[1][i], len(cc[1][0])-1)
#     for j in range(len(cc[1][0])):# for each point in a cluster
#         predVal = cc[1][i][len(cc[1][i])-1]
#         # plt.plot([day],[predVal],marker=".")
#         day = day + 1



for i in range(len(tree)):
    plt.plot(extractColumn(tree[i],0))
plt.show()

# print("\n\n\n tree:  ", tree)
# print(len(tree))
# print(np.arange(100))
# clstr1 = []
# for i in range(len(cc[1][1])):
#     clstr1.append(cc[1][1][i])

# print("cluster1:  ", clstr1)
# plt.scatter(clstr1, np.arange(len(clstr1)))
# plt.show()
# print("\n\n",len(clstr), len(centr))
# print(euclideanDistance([1,2,3],[1,4,5]))
# print(threshold([[1,0,4,2],[0,2,1,4]]))
# print(difference([1,2,3,4],[5,6,7,8]))
# print(diffInCentriod([[1,2,4,5],[5,2,1,7]],[[4,2,7,5],[9,2,0,1]]))
# print(vectorAvg([[1, 2, 4, 5],[5,2,1,7],[3,4,1,6]]))
# print(clusterAvg([[[1, 2, 4, 5],[5,2,1,7]],[[3,4,1,6]]]))
# print(addToCluster([4,2,1,0],[[]],0))
# print(y[0])
# print(initCluster(T,3))
# print(centriod([1,7,6,5],[3,9,7,5]))
# print(similarity([2,3,4],[5,2,1]))

