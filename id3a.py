
import numpy as np
import math


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
        target = extractColumn(T,C-1)
        stdParent = stdDeviation(target)
        # print("Parent Std = ",stdParent)
        mx = -9999
        mxIndex = -1
        for j in range(C-1):
            # print("Column Comparison = ",j)
            uniqueClass = np.unique(extractColumn(T,j))
            # print(uniqueClass)
            lstOfStd = []
            lstOfCnt = []
            for val in uniqueClass:
                # print("Unique Val = ",val)
                subtable = extractTable(T,val,j)
                #print("Subtable = ",subtable,"index and Val ",j,val)
                lst = extractColumn(subtable,len(subtable[0])-1)
                #print("List = ",lst)
                stdChild = (stdDeviation(lst))
                lstOfStd.append(stdChild)
                lstOfCnt.append(len(lst))
            wt = weightedAvg(lstOfStd,lstOfCnt)
            diff = stdParent - wt
            # print("Difference = ", diff)
            if mx < diff:
                mx = diff
                mxIndex = j
        # print("SplittingMaxIndex = ",mxIndex)
        newTestData = removeItem(testData,mxIndex)
        uniqueClass = np.unique(extractColumn(T,mxIndex))
        flag = 0
        for val in uniqueClass:
            if testData[mxIndex] == val:
                flag = 1
                t = extractTable(T,val,mxIndex)
                # print("recursiveCall...")
                return recursiveCall(t,newTestData)
        if flag == 0:
            res = normalAvg(extractColumn(T,len(T[0])-1))
            # print("Test Data = ", testData)
            error = abs(res - testData[len(testData)-1])
            return res,error
            


# ROWS = len(T)

#print(T[0][3]) 
T = [[0,2,1,0,25],[0,2,1,1,30],[1,2,1,0,46],[2,1,1,0,45],[2,0,0,0,52],[2,0,0,1,23],[1,0,0,1,43],[0,1,1,0,35],[0,0,0,0,38],[2,1,0,0,46],[0,1,0,1,48],[1,1,1,1,52],[1,2,0,0,44],[2,1,1,1,30]]
#y = [25, 30, 46, 35, 38, 48]
#print(np.std(y))
# s = 0 
# for i in T:
#     print(recursiveCall(T,i))

def id3a_main(T):
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
# if __name__ == "__id3a_main__":
#     file = open("mainTable1.csv")
#     T = np.loadtxt(file, delimiter=",")
#     # T = [[0,2,1,0,25],[0,2,1,1,30],[1,2,1,0,46],[2,1,1,0,45],[2,0,0,0,52],[2,0,0,1,23],[1,0,0,1,43],[0,1,1,0,35],[0,0,0,0,38],[2,1,0,0,46],[0,1,0,1,48],[1,1,1,1,52],[1,2,0,0,44],[2,1,1,1,30]]
#     id3a_main(T)

#     s = s + y[1]
# print(s/ROWS)
print(id3a_main(T))
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

