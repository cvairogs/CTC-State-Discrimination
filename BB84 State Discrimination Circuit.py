import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import rc
from decimal import *

getcontext().prec = 1

zero = np.matrix([[1.0], [0.0]])
one = np.matrix([[0.0], [1.0]])
plus = np.matrix([[1/math.sqrt(2)], [1/math.sqrt(2)]])
minus = np.matrix([[1/math.sqrt(2)], [-1/math.sqrt(2)]])
inputBasisVectors = [zero, one]

##helper functions
##==================
def outerP(ket):
    factor1 = ket
    factor2 = ket.T
    return factor1*factor2

def zeroMatrixArray(r, c):
    matList = []
    for i in range(r):
        matList.append([])
        for j in range(c):
            matList[i].append(0.0)
    return matList

def identityMatrix(N):
    returnMatrixArray = zeroMatrixArray(N, N)
    for i in range(N):
        returnMatrixArray[i][i] = 1.0
    returnMatrix = np.matrix(returnMatrixArray)
    return returnMatrix        

def t2P(matrix1, matrix2):
    rows1 = matrix1.shape[0]
    cols1 = matrix1.shape[1]
    rows2 = matrix2.shape[0]
    cols2 = matrix2.shape[1]

    cols = cols1*cols2
    rows = rows1*rows2

    returnMatrixArray = zeroMatrixArray(rows, cols)
    
    for i in range(rows1):
        for j in range(cols1):
            for k in range(rows2):
                for l in range(cols2):
                    r = i*rows2+k
                    c = j*cols2+l
                    returnMatrixArray[r][c]=matrix1[i, j]*matrix2[k, l]

    returnMatrix = np.matrix(returnMatrixArray)
    return returnMatrix                

def tensorP(*args):
    if len(args)>2:
        product = t2P(args[0], args[1])
        i=1
        while(i<len(args)-1):
            product = t2P(product, args[i+1])
            i+=1
        return product
    elif len(args)>1: 
        product = t2P(args[0], args[1])
        return product
    else:
        return args[0]

def trace(matrix):
    Sum = 0.0
    for i in range(matrix.shape[0]):
        Sum+=matrix[i,i]
    return Sum

def excise(matrix, r1, r2, c1, c2):
    returnMatrixArray = zeroMatrixArray(r2-r1+1, c2-c1+1)
    for i in range(r2-r1+1):
        for j in range(c2-c1+1):
            returnMatrixArray[i][j] = matrix[r1+i, c1+j]
    returnMatrix = np.matrix(returnMatrixArray)
    return returnMatrix

##def partialTrace(tensor): #particular to this case
##    returnMatrixArray = [[trace(excise(tensor, 0, 3, 0, 3)), trace(excise(tensor, 4, 7, 0, 3))],
##                    [trace(excise(tensor, 4, 7, 0, 3)), trace(excise(tensor, 4, 7, 4, 7))]]
##    returnMatrix = np.matrix(returnMatrixArray)
##    return returnMatrix

def partialTrace(tensor):
    dimension=2
    tRows = tensor.shape[0]
    tCols = tensor.shape[1]
    returnMatrixArray = []
    for i in range(dimension):
        returnMatrixArray.append([])
        for j in range(dimension):
            returnMatrixArray[i].append(trace(excise(tensor, int(i*tRows/dimension), int((i+1)*tRows/dimension-1), int(j*tCols/dimension), int((j+1)*tCols/dimension-1))))
    returnMatrix = np.matrix(returnMatrixArray)
    return returnMatrix

HAD = 1/math.sqrt(2)*np.matrix([[1,1],[1,-1]])
SWAP = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
CHAD = tensorP(outerP(zero), identityMatrix(2))+tensorP(outerP(one), HAD)
CHAD1 = tensorP(identityMatrix(2), outerP(zero))+tensorP(HAD, outerP(one))
X = np.matrix([[0,1],[1,0]])

def SWAPE(pos1, pos2):
    tensorPTemplate = [identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2)]
    tensorPTemplate[pos1] = inputBasisVectors[0]*inputBasisVectors[0].getH()
    tensorPTemplate[pos2] = inputBasisVectors[0]*inputBasisVectors[0].getH()
    returnSWAP = tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
    
    tensorPTemplate[pos1] = inputBasisVectors[0]*inputBasisVectors[1].getH()
    tensorPTemplate[pos2] = inputBasisVectors[1]*inputBasisVectors[0].getH()
    returnSWAP += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[pos1] = inputBasisVectors[1]*inputBasisVectors[0].getH()
    tensorPTemplate[pos2] = inputBasisVectors[0]*inputBasisVectors[1].getH()
    returnSWAP += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[pos1] = inputBasisVectors[1]*inputBasisVectors[1].getH()
    tensorPTemplate[pos2] = inputBasisVectors[1]*inputBasisVectors[1].getH()
    returnSWAP += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
    
    return returnSWAP

def u00(indexList):
    S1 = indexList[0]
    S2 = indexList[1]
    C1 = indexList[2]
    C2 = indexList[3]
    
    tensorPTemplate = [identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2)]
    tensorPTemplate[S1] = outerP(zero)
    tensorPTemplate[S2] = outerP(zero)
    
    tensorPTemplate[C1] = inputBasisVectors[0]*inputBasisVectors[0].getH() 
    tensorPTemplate[C2] = inputBasisVectors[0]*inputBasisVectors[0].getH() 
    returnU00 = tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[C1] = inputBasisVectors[1]*inputBasisVectors[0].getH()
    tensorPTemplate[C2] = inputBasisVectors[0]*inputBasisVectors[1].getH()
    returnU00 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[C1] = inputBasisVectors[0]*inputBasisVectors[1].getH()
    tensorPTemplate[C2] = inputBasisVectors[1]*inputBasisVectors[0].getH()
    returnU00 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[C1] = inputBasisVectors[1]*inputBasisVectors[1].getH()
    tensorPTemplate[C2] = inputBasisVectors[1]*inputBasisVectors[1].getH()
    returnU00 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(zero) 
    tensorPTemplate[S2] = outerP(one)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU00 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(one) 
    tensorPTemplate[S2] = outerP(zero)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU00 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(one) 
    tensorPTemplate[S2] = outerP(one)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU00 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
    
    return returnU00

def u01(indexList):
    S1 = indexList[0]
    S2 = indexList[1]
    C1 = indexList[2]
    C2 = indexList[3]

    tensorPTemplate = [identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2)]
    tensorPTemplate[S1] = outerP(zero)
    tensorPTemplate[S2] = outerP(one)
    tensorPTemplate[C1] = X
    tensorPTemplate[C2] = X
    returnU01 = tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(zero) 
    tensorPTemplate[S2] = outerP(zero)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU01 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(one) 
    tensorPTemplate[S2] = outerP(zero)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU01 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(one) 
    tensorPTemplate[S2] = outerP(one)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU01 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])    
    return returnU01

def u10(indexList):
    S1 = indexList[0]
    S2 = indexList[1]
    C1 = indexList[2]
    C2 = indexList[3]

    tensorPTemplate = [identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2)]
    tensorPTemplate[S1] = outerP(one)
    tensorPTemplate[S2] = outerP(zero)
    tensorPTemplate[C1] = X*HAD
    tensorPTemplate[C2] = identityMatrix(2)*identityMatrix(2)
    returnU10 = tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(zero) 
    tensorPTemplate[S2] = outerP(zero)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU10 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(zero) 
    tensorPTemplate[S2] = outerP(one)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU10 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(one) 
    tensorPTemplate[S2] = outerP(one)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU10 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
    
    return returnU10

def u11(indexList):
    S1 = indexList[0]
    S2 = indexList[1]
    C1 = indexList[2]
    C2 = indexList[3]

    tensorPTemplate = [identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2)]

    tensorPTemplate[S1] = identityMatrix(2)
    tensorPTemplate[S2] = identityMatrix(2)
    tensorPTemplate[C1] = X
    tensorPTemplate[C2] = HAD
    returnU11 = tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(one) 
    tensorPTemplate[S2] = outerP(one)
    
    tensorPTemplate[C1] = inputBasisVectors[0]*inputBasisVectors[0].getH()
    tensorPTemplate[C2] = inputBasisVectors[0]*inputBasisVectors[0].getH()
    returnU11SUB = tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
    
    tensorPTemplate[C1] = inputBasisVectors[0]*inputBasisVectors[1].getH()
    tensorPTemplate[C2] = inputBasisVectors[1]*inputBasisVectors[0].getH()
    returnU11SUB += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[C1] = inputBasisVectors[1]*inputBasisVectors[0].getH()
    tensorPTemplate[C2] = inputBasisVectors[0]*inputBasisVectors[1].getH()
    returnU11SUB += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[C1] = inputBasisVectors[1]*inputBasisVectors[1].getH()
    tensorPTemplate[C2] = inputBasisVectors[1]*inputBasisVectors[1].getH()
    returnU11SUB += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
    returnU11*=returnU11SUB
    

    tensorPTemplate[S1] = outerP(zero) 
    tensorPTemplate[S2] = outerP(zero)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU11 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(one) 
    tensorPTemplate[S2] = outerP(zero)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU11 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])

    tensorPTemplate[S1] = outerP(zero) 
    tensorPTemplate[S2] = outerP(one)
    tensorPTemplate[C1] = identityMatrix(2)
    tensorPTemplate[C2] = identityMatrix(2)
    returnU11 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
    
    return returnU11

Nq = 2
gamma = 0
registerVector = np.matrix([[float(math.sqrt(gamma))], [float(math.sqrt(1-gamma))]])
inputStates = [zero, one, plus, minus]
inputStatesProbWeights = [1,0,0,0]#[.25, .25, .25, .25]#[1,0,0,0]

#inputIndex=1 
rhoF = outerP(registerVector)
#rhoS1 = outerP(inputStates[inputIndex])
rhoS2 = outerP(zero)
#rhoC1 = .5*outerP(zero) + .5*outerP(one)
#rhoC2 = .5*outerP(zero) + .5*outerP(one)
pi = .5*outerP(zero) + .5*outerP(one)

####GATE TESTING
####=================================================
##print("Initial SWAPS")
##print("")
##print("rhoS1 before: ")
##print(rhoS1)
##print("rhoS2 before: ")
##print(rhoS2)
##print("rhoC1 before: ")
##print(rhoC1)
##print("rhoC2 before: ")
##print(rhoC2)
##print("")
##
##rho = tensorP(rhoS1,rhoS2,rhoC1,rhoC2)
##U=u11([0,1,2,3])*u10([0,1,2,3])*u01([0,1,2,3])*u00([0,1,2,3])*SWAPE(1,3)*SWAPE(0,2)
##S1 = partialTrace(U*rho*U.getH())
##
##rho1 = tensorP(rhoS2,rhoS1,rhoC1,rhoC2)
##U=u11([1,0,2,3])*u10([1,0,2,3])*u01([1,0,2,3])*u00([1,0,2,3])*SWAPE(0,3)*SWAPE(1,2)
##S2 = partialTrace(U*rho1*U.getH())
##
##rho2 = tensorP(rhoC1,rhoS1,rhoS2,rhoC2)
##U=u11([1,2,0,3])*u10([1,2,0,3])*u01([1,2,0,3])*u00([1,2,0,3])*SWAPE(2,3)*SWAPE(0,1)
##C1 = partialTrace(U*rho2*U.getH())
##
##rho3 = tensorP(rhoC2,rhoS1,rhoS2,rhoC1)
##U=u11([1,2,3,0])*u10([1,2,3,0])*u01([1,2,3,0])*u00([1,2,3,0])*SWAPE(0,2)*SWAPE(1,3)
##C2 = partialTrace(U*rho3*U.getH())
##
##rhoS1 = S1
##rhoS2 = S2
##rhoC1 = C1
##rhoC2 = C2
##
##print("rhoS1 after: ")
##print(S1)
##print("rhoS2 after: ")
##print(S2)
##print("rhoC1 after: ")
##print(C1)
##print("rhoC2 after: ")
##print(C2)
##print("================================================")
##
##print("u00")
##print("")
##print("rhoS1 before: ")
##print(S1)
##print("rhoS2 before: ")
##print(S2)
##print("rhoC1 before: ")
##print(C1)
##print("rhoC2 before: ")
##print(C2)
##print("")
##
##rho = tensorP(rhoS1,rhoS2,rhoC1,rhoC2)
##U=u00([0,1,2,3])
##S1 = partialTrace(U*rho*U.getH())
##
##rho1 = tensorP(rhoS2,rhoS1,rhoC1,rhoC2)
##U=u00([1,0,2,3])
##S2 = partialTrace(U*rho1*U.getH())
##
##rho2 = tensorP(rhoC1,rhoS1,rhoS2,rhoC2)
##U=u00([1,2,0,3])
##C1 = partialTrace(U*rho2*U.getH())
##
##rho3 = tensorP(rhoC2,rhoS1,rhoS2,rhoC1)
##U=u00([1,2,3,0])
##C2 = partialTrace(U*rho3*U.getH())
##
##rhoS1 = S1
##rhoS2 = S2
##rhoC1 = C1
##rhoC2 = C2
##
##print("rhoS1 after: ")
##print(S1)
##print("rhoS2 after: ")
##print(S2)
##print("rhoC1 after: ")
##print(C1)
##print("rhoC2 after: ")
##print(C2)
##print("================================================")
##
##print("u01")
##print("")
##print("rhoS1 before: ")
##print(S1)
##print("rhoS2 before: ")
##print(S2)
##print("rhoC1 before: ")
##print(C1)
##print("rhoC2 before: ")
##print(C2)
##print("")
##
##rho = tensorP(rhoS1,rhoS2,rhoC1,rhoC2)
##U=u01([0,1,2,3])
##S1 = partialTrace(U*rho*U.getH())
##
##rho1 = tensorP(rhoS2,rhoS1,rhoC1,rhoC2)
##U=u01([1,0,2,3])
##S2 = partialTrace(U*rho1*U.getH())
##
##rho2 = tensorP(rhoC1,rhoS1,rhoS2,rhoC2)
##U=u01([1,2,0,3])
##C1 = partialTrace(U*rho2*U.getH())
##
##rho3 = tensorP(rhoC2,rhoS1,rhoS2,rhoC1)
##U=u01([1,2,3,0])
##C2 = partialTrace(U*rho3*U.getH())
##
##rhoS1 = S1
##rhoS2 = S2
##rhoC1 = C1
##rhoC2 = C2
##
##print("rhoS1 after: ")
##print(S1)
##print("rhoS2 after: ")
##print(S2)
##print("rhoC1 after: ")
##print(C1)
##print("rhoC2 after: ")
##print(C2)
##print("================================================")
##
##print("u10")
##print("")
##print("rhoS1 before: ")
##print(S1)
##print("rhoS2 before: ")
##print(S2)
##print("rhoC1 before: ")
##print(C1)
##print("rhoC2 before: ")
##print(C2)
##print("")
##
##rho = tensorP(rhoS1,rhoS2,rhoC1,rhoC2)
##U=u10([0,1,2,3])
##S1 = partialTrace(U*rho*U.getH())
##
##rho1 = tensorP(rhoS2,rhoS1,rhoC1,rhoC2)
##U=u10([1,0,2,3])
##S2 = partialTrace(U*rho1*U.getH())
##
##rho2 = tensorP(rhoC1,rhoS1,rhoS2,rhoC2)
##U=u10([1,2,0,3])
##C1 = partialTrace(U*rho2*U.getH())
##
##rho3 = tensorP(rhoC2,rhoS1,rhoS2,rhoC1)
##U=u10([1,2,3,0])
##C2 = partialTrace(U*rho3*U.getH())
##
##rhoS1 = S1
##rhoS2 = S2
##rhoC1 = C1
##rhoC2 = C2
##
##print("rhoS1 after: ")
##print(S1)
##print("rhoS2 after: ")
##print(S2)
##print("rhoC1 after: ")
##print(C1)
##print("rhoC2 after: ")
##print(C2)
##print("================================================")
##
##print("u11")
##print("")
##print("rhoS1 before: ")
##print(S1)
##print("rhoS2 before: ")
##print(S2)
##print("rhoC1 before: ")
##print(C1)
##print("rhoC2 before: ")
##print(C2)
##print("")
##
##rho = tensorP(rhoS1,rhoS2,rhoC1,rhoC2)
##U=u11([0,1,2,3])
##S1 = partialTrace(U*rho*U.getH())
##
##rho1 = tensorP(rhoS2,rhoS1,rhoC1,rhoC2)
##U=u11([1,0,2,3])
##S2 = partialTrace(U*rho1*U.getH())
##
##rho2 = tensorP(rhoC1,rhoS1,rhoS2,rhoC2)
##U=u11([1,2,0,3])
##C1 = partialTrace(U*rho2*U.getH())
##
##rho3 = tensorP(rhoC2,rhoS1,rhoS2,rhoC1)
##U=u11([1,2,3,0])
##C2 = partialTrace(U*rho3*U.getH())
##
##print("rhoS1 after: ")
##print(S1)
##print("rhoS2 after: ")
##print(S2)
##print("rhoC1 after: ")
##print(C1)
##print("rhoC2 after: ")
##print(C2)
##print("================================================")



#indices = [S1, S2, C1, C2]
indices0 = [0, 1, 2, 3]
indices1 = [1, 0, 2, 3]
indices2 = [2, 1, 0, 3]
indices3 = [3, 1, 2, 0]

#0. rho = rhoS1 * rhoS2 * rhoC1 * rhoC2 * rhoF

U0 = tensorP(identityMatrix(16), outerP(zero)) + tensorP(u11(indices0)*u10(indices0)*u01(indices0)*u00(indices0)*SWAPE(1,3)*SWAPE(0,2), outerP(one))

#1. rho = rhoS2 * rhoS1 * rhoC1 * rhoC2 * rhoF

U1 = tensorP(identityMatrix(16), outerP(zero)) + tensorP(u11(indices1)*u10(indices1)*u01(indices1)*u00(indices1)*SWAPE(0,3)*SWAPE(1,2), outerP(one))

#2. rho = rhoC1 * rhoS2 * rhoS1 * rhoC2 * rhoF

U2 = tensorP(identityMatrix(16), outerP(zero)) + tensorP(u11(indices2)*u10(indices2)*u01(indices2)*u00(indices2)*SWAPE(1,3)*SWAPE(0,2), outerP(one))

#3. rho = rhoC2 * rhoS2 * rhoC1 * rhoS1 * rhoF

U3 = tensorP(identityMatrix(16), outerP(zero)) + tensorP(u11(indices3)*u10(indices3)*u01(indices3)*u00(indices3)*SWAPE(0,1)*SWAPE(2,3), outerP(one))

#successProb = [zero, one, plus, minus]
successProb = [0, 0, 0, 0]
#rhoC1Out = [zero, one, plus, minus]
rhoC1List = [pi, pi, pi, pi]
#rhoC2Out = [zero, one, plus, minus]
rhoC2List= [pi, pi, pi, pi]

for i in range(Nq):

    for j in range(1):
        inputIndex = j

        rhoS1 = outerP(inputStates[inputIndex])
        rhoC1 = rhoC1List[inputIndex]
        rhoC2 = rhoC2List[inputIndex]

        
        #0. rho = rhoS1 * rhoS2 * rhoC1 * rhoC2 * rhoF
        
        rho0 = tensorP(rhoS1, rhoS2, rhoC1, rhoC2, rhoF)

        #1. rho = rhoS2 * rhoS1 * rhoC1 * rhoC2 * rhoF
        
        rho1 = tensorP(rhoS2, rhoS1, rhoC1, rhoC2, rhoF)

        #2. rho = rhoC1 * rhoS2 * rhoS1 * rhoC2 * rhoF

        rho2 = tensorP(rhoC1, rhoS2, rhoS1, rhoC2, rhoF)

        #3. rho = rhoC2 * rhoS2 * rhoC1 * rhoS1 * rhoF

        rho3 = tensorP(rhoC2, rhoS2, rhoC1, rhoS1, rhoF)
            
        transRho0 = U0*rho0*U0.getH()
        transRho1 = U1*rho1*U1.getH()
        transRho2 = U2*rho2*U2.getH()
        transRho3 = U3*rho3*U3.getH()

        

        
        rhoS1Out = partialTrace(transRho0)
        rhoS2Out = partialTrace(transRho1)
        rhoC1Out = partialTrace(transRho2)
        rhoC2Out = partialTrace(transRho3)

        if(i==0):
            print(tensorP(rhoC1, rhoC2))

        print("Tr: ")
        print("======================")
##        print(trace(rho0))
##        print(trace(transRho0))
##        newRho = tensorP(rhoS1, rhoS2, rhoC1Out, rhoC2Out)
##        print(trace(newRho))
        print("N: " + str(i+1))
        print(tensorP(rhoC2Out))
        print("======================")

        """
        rhoS1Out *= 1/trace(rhoS1Out)
        rhoS2Out *= 1/trace(rhoS2Out)
        rhoC1Out *= 1/trace(rhoC1Out)
        rhoC2Out *= 1/trace(rhoC2Out)

        """

        

        if (inputIndex == 0):
            s = rhoS1Out[0,0]*rhoS2Out[0,0]
##            print("zero")
##            print(rhoS1Out)
##            print(rhoS2Out)
##            print(rhoC1Out)
##            print(rhoC2Out)
        elif (inputIndex == 1):
            s = rhoS1Out[0,0]*rhoS2Out[1,1]
            #print("one")
            #print(trace(rhoS1Out))
        elif (inputIndex == 2):
            s = rhoS1Out[1,1]*rhoS2Out[0,0]
            #print("minus")
            #print(trace(rhoS1Out))
        elif (inputIndex == 3):
            s = rhoS1Out[1,1]*rhoS2Out[1,1]
            #print("plus")
            #print(trace(rhoS1Out))
        else:
            print("ERROR")

        rhoC1List[inputIndex] = rhoC1Out
        rhoC2List[inputIndex] = rhoC2Out
        successProb[inputIndex] = s
        

    averageSuccessProb = inputStatesProbWeights[0]*successProb[0] + inputStatesProbWeights[1]*successProb[1] + inputStatesProbWeights[2]*successProb[2] + inputStatesProbWeights[3]*successProb[3]
    N=i+1

    #print(averageSuccessProb)
    
    plt.scatter(N, averageSuccessProb, s= 30, c= 'r', alpha = 0.5)

plt.xlabel("N")
plt.ylabel("Success Prob.")
plt.show()

