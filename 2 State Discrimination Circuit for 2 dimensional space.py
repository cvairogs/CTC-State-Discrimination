import numpy as np
import matplotlib.pyplot as plt
import math
import random

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

def norm(ket):
    return math.sqrt(ket[0,0]**2 + ket[1,0]**2)

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

M = 2 #dimension of vector space

inputBasisVectors = []
for i in range(M):
    inputBasisVectors.append(excise(identityMatrix(M), 0, M-1, i, i))
                        
inputStates = [math.sqrt(7/8)*zero + math.sqrt(1/8)*one, math.sqrt(1/4)*zero + math.sqrt(3/4)*one]


unitaries = []

#for now we assume that input states are from two dimensional vector space.

for k in range(M):
    cmBasis = []
    bmBasis = []
    #b1
    bmBasis.append(inputStates[k])

    #c1
    cmBasis.append(inputBasisVectors[k])

    #b2
    index = random.randint(0, len(inputStates)-1)
    while (k==index):
        index = random.randint(0, len(inputStates)-1)
    vector = inputStates[index] - outerP(bmBasis[0])*inputStates[index]
    vector = vector/norm(vector)
    bmBasis.append(vector)

    #c2
    Sum = np.matrix(zeroMatrixArray(M, 1))
    counter = 0
    for i in range(len(inputBasisVectors)):    
        if(i!=k):
            Sum = Sum + inputBasisVectors[i]
            counter+=1
    cmBasis.append(1/math.sqrt(counter)*Sum)

##    print(k)
##    print(cmBasis)
##    print(bmBasis)

    U=np.matrix(zeroMatrixArray(M, M))
    for j in range(M):
        U=U+cmBasis[j]*bmBasis[j].getH()
    unitaries.append(U)

print(unitaries)
print(inputStates)
print(inputBasisVectors)


Nq = 200
gamma = 0
registerVector = np.matrix([[math.sqrt(gamma)], [math.sqrt(1-gamma)]])


HAD = 1/math.sqrt(2)*np.matrix([[1,1],[1,-1]])
SWAP = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
CHAD = tensorP(outerP(zero), identityMatrix(2))+tensorP(outerP(one), HAD)
CHAD1 = tensorP(identityMatrix(2), outerP(zero))+tensorP(HAD, outerP(one))

rhoF = outerP(registerVector)
rhoS = outerP(inputStates[0])
rhoC = .5*outerP(zero) + .5*outerP(one)

#U = tensorP(identityMatrix(4), outerP(zero)) + tensorP(CHAD*SWAP, outerP(one))
#U1 = tensorP(identityMatrix(4), outerP(zero)) + tensorP(CHAD1*SWAP, outerP(one))
CU = tensorP(outerP(inputBasisVectors[0]), unitaries[0])
CU1 = tensorP(unitaries[0], outerP(inputBasisVectors[0]))
for i in range(1, M):
    CU = CU + tensorP(outerP(inputBasisVectors[i]), unitaries[i])
    CU1 = CU1 + tensorP(unitaries[i], outerP(inputBasisVectors[i]))
    #print("A: ")
    #print(unitaries[i]*inputStates[i] - inputBasisVectors[i])

U = tensorP(identityMatrix(4), outerP(zero)) + tensorP(CU*SWAP, outerP(one))
U1 = tensorP(identityMatrix(4), outerP(zero)) + tensorP(CU1*SWAP, outerP(one))

for i in range(Nq):
    rho = tensorP(rhoS, rhoC, rhoF)
    rho1 = tensorP(rhoC, rhoS, rhoF)
    transRho = U*rho*U.getH()
    transRho1 = U1*rho1*U1.getH()
    rhoSOut = partialTrace(transRho)
    rhoC = partialTrace(transRho1)
    #print(rhoSOut)
    epsilon = rhoSOut[1, 1]
    plt.scatter(i+1, epsilon, s= 30, c= 'r', alpha = 0.5)
    #print(epsilon)
##x=np.arange(1.0, Nq, 0.02)
#plt.plot(x, 0.5*gamma + (1+gamma)**(x-1)*(1-gamma)/2**x, 'b--')


plt.xlabel("N")
plt.ylabel("Error Prob.")
plt.show()


