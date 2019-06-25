import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import rc
import random
from collections import Counter
from random import choices

zero = np.matrix([[1], [0]])
one = np.matrix([[0], [1]])
plus = np.matrix([[1/math.sqrt(2)], [1/math.sqrt(2)]])
minus = np.matrix([[1/math.sqrt(2)], [-1/math.sqrt(2)]])

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
            matList[i].append(0)
    return matList

def identityMatrix(N):
    returnMatrixArray = zeroMatrixArray(N, N)
    for i in range(N):
        returnMatrixArray[i][i] = 1
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
    Sum = 0
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

def partialTrace(tensor): #particular to this case
    returnMatrixArray = [[trace(excise(tensor, 0, 3, 0, 3)), trace(excise(tensor, 4, 7, 0, 3))],
                    [trace(excise(tensor, 4, 7, 0, 3)), trace(excise(tensor, 4, 7, 4, 7))]]
    returnMatrix = np.matrix(returnMatrixArray)
    return returnMatrix

Nq = 2000
gamma = .9
registerVector = np.matrix([[math.sqrt(gamma)], [math.sqrt(1-gamma)]])
inputStates = [zero, minus]
inputStatesProbWeights = [0, 1]

HAD = 1/math.sqrt(2)*np.matrix([[1,1],[1,-1]])
SWAP = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
CHAD = tensorP(outerP(zero), identityMatrix(2))+tensorP(outerP(one), HAD)
CHAD1 = tensorP(identityMatrix(2), outerP(zero))+tensorP(HAD, outerP(one))

rhoF = outerP(registerVector)

U = tensorP(identityMatrix(4), outerP(zero)) + tensorP(CHAD*SWAP, outerP(one))
U1 = tensorP(identityMatrix(4), outerP(zero)) + tensorP(CHAD1*SWAP, outerP(one))

initCTCProbWeights = [1, 0]
initCTCState = initCTCProbWeights[0]*outerP(zero) + initCTCProbWeights[1]*outerP(one)

rhoCList = [initCTCState, initCTCState]


for i in range(Nq):
    pN = 0
    
    for j in range(2):
        inputIndex = j
        
        rhoS = outerP(inputStates[inputIndex])
        rhoC = rhoCList[inputIndex]
        
        rho = tensorP(rhoS, rhoC, rhoF)
        rho1 = tensorP(rhoC, rhoS, rhoF)
        
        transRho = U*rho*U.getH()
        transRho1 = U1*rho1*U1.getH()
        
        rhoSOut = partialTrace(transRho)
        rhoCOut = partialTrace(transRho1)
        
        rhoCList[inputIndex] = rhoCOut
        
        if (inputIndex == 0):
            epsilon = rhoCOut[0,0]
            print("input state was zero")
            print("probability of measuring zero was: " + str(epsilon))
        if (inputIndex == 1):
            epsilon = rhoCOut[1,1]
            print("input state was minus")
            print("probability of measuring one was: " + str(epsilon))
        

        pN += epsilon*inputStatesProbWeights[inputIndex]
        
    plt.scatter(i+1, math.log(1-pN), s= 30, c= 'r', alpha = 0.5)
    #P = (1-initCTCProbWeights[1]*(1+gamma)**(i+1)/2**(i+1))*inputStatesProbWeights[0] + (1-initCTCProbWeights[0]*(1+gamma)**(i+1)/2**(i+1))*inputStatesProbWeights[1]
    #plt.scatter(i+1, P, s= 30, c = 'b', alpha = 0.5)
    
x=np.arange(1.0, Nq, 0.02)
#plt.plot(x, 0.5*gamma + (1+gamma)**(x-1)*(1-gamma)/2**x, 'b--')


plt.xlabel("N")
#plt.ylabel("Error Prob.")
plt.show()

