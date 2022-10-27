import numpy as np
import math
import matplotlib.pyplot as plt


def weightedAvg(v,c):
    N = len(v)
    s = 0
    s1 = 0
    for i in range(N):
        s = s + v[i]*c[i]
    for i in c:
        s1 = s1 + i
    return s/s1

def extractTable(T,val,index):
    R = len(T)
    C = len(T[0])
    subtable = []
    for i in range(R):
        if T[i][index] == val:
            newRow = []
            for j in range(C):
                if j != index:
                    newRow.append(T[i][j])
            subtable.append(newRow)
    return subtable

def extractColumn(T,index):
    R = len(T)
    col = []
    for i in range(R):
        col.append(T[i][index])
    return col

def normalAvg(x):
    N = len(x)
    s = 0
    for i in range(N):
        s = s + x[i]
    return s/N

def varianceData(x):
    u = normalAvg(x)
    N = len(x)
    s = 0
    for i in range(N):
        s = s + (x[i]-u)*(x[i]-u)
    return s/N

def stdDeviation(x):
    return math.sqrt(varianceData(x))

def removeItem(List,index):
    tmp = []
    for i in range(len(List)):
        if index != i:
            tmp.append(List[i])
    return tmp

def cntElementClass(T,index):# [0,1,4,1,4] == > [0,1,4]
    res = []# [[0,1],[1,2],[4,2]]
    List = extractColumn(T,index) # [0,1,4,1,4], [0,1,4][1, 2, 2]
    res, count = np.unique(extractColumn(T,index), return_counts = True )# [0,1,4]
    numElements = len(List)
    return res, numElements, count

def giniImpurity(x):# [3,1,2]==> P(i)*(1-P(i))
    s = 0
    N = len(x)
    total = normalAvg(x)*N
    for i in range(len(x)):
        s = s + (x[i]/total)*(1-(x[i]/total))
    return s

def entropy(x):# -SUM(p log(p))
    s = 0
    N = len(x)
    total = normalAvg(x)*N
    for i in range(len(x)):
        if x[i] != 0:
            s = s + (x[i]/total)*math.log2(x[i]/total)
    return -s

def recursiveCall(T,testData):
    # print("Table = ",T)
    R = len(T)
    C = len(T[0])
    if C <= 2:
        # print("base case... terminating!!!")
        # print("Base Case Table = " ,T)
        # print("Test Data = ", testData)
        res = normalAvg(extractColumn(T,C-1))
        error = abs(res - testData[len(testData)-1])
        # print("by avg")
        return res,error
    else:
#        y = cntElementClass(T,C-1)
#        parentGini = giniImpurity(y[2])
#        print("Parent Std = ",parentGini)
        mn = 99999
        mnIndex = -1
        for j in range(C-1):
            y = cntElementClass(T,j)
            childGini = giniImpurity(y[2])
            if childGini < mn:
                mn = childGini
                mnIndex = j
        # print("SplittingMaxIndex = ",mxIndex)
        newTestData = removeItem(testData,mnIndex)
        uniqueClass = np.unique(extractColumn(T,mnIndex))
        flag = 0
        for val in uniqueClass:
            if testData[mnIndex] == val:
                flag = 1
                t = extractTable(T,val,mnIndex)
                # print("recursiveCall...")
                return recursiveCall(t,newTestData)
        if flag == 0:
            res = normalAvg(extractColumn(T,len(T[0])-1))
            # print("Test Data = ", testData)
            error = abs(res - testData[len(testData)-1])
            return res,error
            
# file = open("mainTable0.csv")
file = open("covid_updated_file.csv")
T = np.loadtxt(file, delimiter=",")
ROWS = len(T)
# targetValues = np.loadtxt(open("/content/mainTable0.csv", "rb"), delimiter=",", usecols=3)
targetValues = np.loadtxt(open("covid_updated_file.csv", "rb"), delimiter=",", usecols=6)


def CART_main(T):
    ROWS = len(T)
    print(ROWS)
    res = []
    print("----------------------------------------------------------")
    acc_error = 0
    rmse = []
    for i in T:
        yy = (recursiveCall(T,i))
        acc_error = acc_error + (yy[1])*(yy[1])
        acc_error =  acc_error/2
        rmse.append(math.sqrt(acc_error))
    rmseAvg = normalAvg(rmse)
    # print("RMSE  = ",math.sqrt(acc_error))
    print("-----------------------------------------------------------")
    for i in T:
        res.append((recursiveCall(T,i)))
    return res, rmseAvg

#print(T[0][3]) 
#T = [[0,2,1,0,25],[0,2,1,1,30],[1,2,1,0,46],[2,1,1,0,45],[2,0,0,0,52],[2,0,0,1,23],[1,0,0,1,43],[0,1,1,0,35],[0,0,0,0,38],[2,1,0,0,46],[0,1,0,1,48],[1,1,1,1,52],[1,2,0,0,44],[2,1,1,1,30]]
#y = [25, 30, 46, 35, 38, 48]
#print(np.std(y))
print("----------------------------------------------------------")
acc_error = 0
for i in T:
    yy = (recursiveCall(T,i))
    acc_error = acc_error + (yy[1])*(yy[1])
acc_error =  acc_error/2
print("RMSE  = ",math.sqrt(acc_error))
print("-----------------------------------------------------------")

s = 0 
predictions = []
# print(targetValues)
for i in T:
    # print(recursiveCall(T,i))
    predictions.append(recursiveCall(T,i))
predictedValues = []
for pred, err in predictions:
  predictedValues.append(pred)
print(predictedValues)

plt.plot(predictedValues)
# plt.hold(True)
plt.plot(targetValues)
plt.title("CART-Gini Impurity")
plt.legend(["Predicted","Actual"])
plt.xlabel("Number of Days")
plt.ylabel("Number of Cases")
plt.show()
#y = (cntElementClass([[1,2,3],[1,3,5],[2,3,4]],0))
#print(y[2])
#print(entropy([4,4,4,4]))
#     s = s + y[1]
# print(s/ROWS)
#print(max(extractColumn(T,0)))
#y = [[0,6,3],[1,4,3],[2,5,5],[2,6,5],[0,7,6]]
#t = [25,30,46,45,52,23,43,35,38,46,48,52,44,30]
#print(stdDeviation(t))
#print(normalAvg(y[1]))
#print(extractColumn(y,2))
#print(extractColumn(extractTable(T,0,0),3))
#print(weightedAvg([3.49,7.78,10.87],[4,5,5]))
# p = (np.unique([1,1,2,5,6,1,2]))
# print(p[2])
