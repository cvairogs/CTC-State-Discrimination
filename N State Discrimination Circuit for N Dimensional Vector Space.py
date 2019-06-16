import numpy as np
import matplotlib.pyplot as plt
import math
import random

#zero = np.matrix([[1], [0], [0]])
#one = np.matrix([[0], [1]])
#plus = np.matrix([[1/math.sqrt(2)], [1/math.sqrt(2)]])
#minus = np.matrix([[1/math.sqrt(2)], [-1/math.sqrt(2)]])

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
    dimension = ket.shape[0]
    norm2 = 0
    for i in range(dimension):
        norm2+=ket[i,0]**2
    return math.sqrt(norm2)

def excise(matrix, r1, r2, c1, c2):
    returnMatrixArray = zeroMatrixArray(r2-r1+1, c2-c1+1)
    for i in range(r2-r1+1):
        for j in range(c2-c1+1):
            returnMatrixArray[i][j] = matrix[r1+i, c1+j]
    returnMatrix = np.matrix(returnMatrixArray)
    return returnMatrix

def partialTrace(tensor, dimension): #particular to this case
    tRows = tensor.shape[0]
    tCols = tensor.shape[1]
    returnMatrixArray = []
    for i in range(dimension):
        returnMatrixArray.append([])
        for j in range(dimension):
            returnMatrixArray[i].append(trace(excise(tensor, int(i*tRows/dimension), int((i+1)*tRows/dimension-1), int(j*tCols/dimension), int((j+1)*tCols/dimension-1))))
    returnMatrix = np.matrix(returnMatrixArray)
    return returnMatrix

def projection(ket, basis):
    dimension = ket.shape[0]
    returnKet = np.matrix(zeroMatrixArray(dimension, 1))
    for i in range(len(basis)):
        returnKet = returnKet + outerP(basis[i])*ket
    return returnKet

def ketEqual(ket1, ket2):
    result = True
    for i in range(ket1.shape[0]):
        if ket1[i,0] != ket2[i,0]:
            result = False
    return result

def ketApproxEqual(ket1, ket2, epsilon):
    result = True
    for i in range(ket1.shape[0]):
        if abs(ket1[i, 0] - ket2[i, 0])>epsilon:
##            print("two vectors are not equal")
##            print("difference: " + str(ket1[i, 0] - ket2[i, 0]))
##            print("limit: " + str(epsilon))
##            print("ket difference exceeds limit: " + str((ket1[i, 0] - ket2[i, 0])>=ket1[i, 0]*epsilon))
            result = False
            
    return result

M = 3 #dimension of vector space

inputBasisVectors = []
for i in range(M):
    inputBasisVectors.append(excise(identityMatrix(M), 0, M-1, i, i))
                        
inputStates = [np.matrix([[1], [0], [0]]), np.matrix([[math.sqrt(1/12)], [math.sqrt(1/6)], [math.sqrt(2/3)]]), np.matrix([[-1/2], [-1/2], [math.sqrt(1/2)]])]

unitaries = []

#print(ketEqual(np.matrix([[0],[1]]), np.matrix([[0],[1]])))

#for now we assume that input states are from two dimensional vector space.

for k in range(M):
    cmBasis = []
    bmBasis = []
    usedStates = []
    
    #b1
    bmBasis.append(inputStates[k])
    usedStates.append(k)

    #c1
    cmBasis.append(inputBasisVectors[k])

    for l in range(M-1):
        index = random.randint(0, len(inputStates)-1)
        while (index in usedStates):
            index = random.randint(0, len(inputStates)-1)
        outerPSum = np.matrix(zeroMatrixArray(M, M))
        for n in range(len(bmBasis)):
            outerPSum = outerPSum + outerP(bmBasis[n])
        #print(outerPSum)
        vector = inputStates[index] - outerPSum*inputStates[index]
        vector = vector/norm(vector)
        bmBasis.append(vector)
##        print(index)
##        print(bmBasis)
##        print(" ")
       
        
        
        #print(usedStates)
        Sum = np.matrix(zeroMatrixArray(M, 1))
        counter = 1
        for m in range(M):
##            print("projection of input state m: " + str(projection(inputStates[m], bmBasis)))
##            print("input state m: " + str(inputStates[m]))
##            print("the vector may be spanned by the current basis: " + str(ketApproxEqual(projection(inputStates[m], bmBasis), inputStates[m], 10**(-6))))
##            print("m is not used: " + str(not(m in usedStates)))
##            print(" ")
            if (ketApproxEqual(projection(inputStates[m], bmBasis), inputStates[m], 10**(-6)) and not(m in usedStates)):
                Sum = Sum + inputBasisVectors[m]
                counter+=1
                
        cmBasis.append(1/math.sqrt(counter)*Sum)
        usedStates.append(index)
        

##    print(k)
##    print(cmBasis)
##    print(bmBasis)

    U=np.matrix(zeroMatrixArray(M, M))
    for j in range(M):
        U=U+cmBasis[j]*bmBasis[j].getH()
    unitaries.append(U)

##print(unitaries)
##print(inputStates)
##print(inputBasisVectors)


Nq = 40
gamma = 0

#construct registerVector
registerVectorArray = [[math.sqrt(gamma)], [math.sqrt(1-gamma)]]
if (M>2):
    for i in range(2, M):
        registerVectorArray.append([0])
registerVector = np.matrix(registerVectorArray)


##HAD = 1/math.sqrt(2)*np.matrix([[1,1],[1,-1]])
##SWAP = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
##CHAD = tensorP(outerP(zero), identityMatrix(2))+tensorP(outerP(one), HAD)
##CHAD1 = tensorP(identityMatrix(2), outerP(zero))+tensorP(HAD, outerP(one))

#construct SWAP gate
SWAP = np.matrix(zeroMatrixArray(M*M, M*M))
for i in range(M):
    for j in range(M):
        SWAP+=(tensorP(inputBasisVectors[i]*inputBasisVectors[j].getH(), inputBasisVectors[j]*inputBasisVectors[i].getH()))

inputStateIndex = 2
rhoF = outerP(registerVector)
rhoS = outerP(inputStates[inputStateIndex])
#prepare maximally mixed state
rhoC = 1*outerP(inputBasisVectors[0]) + 0*outerP(inputBasisVectors[1]) + 0*outerP(inputBasisVectors[2])
##np.matrix(zeroMatrixArray(M, M))
##for i in range(M):
##    rhoC = rhoC + 1.0/M*outerP(inputBasisVectors[i])

#U = tensorP(identityMatrix(4), outerP(zero)) + tensorP(CHAD*SWAP, outerP(one))
#U1 = tensorP(identityMatrix(4), outerP(zero)) + tensorP(CHAD1*SWAP, outerP(one))
CU = tensorP(outerP(inputBasisVectors[0]), unitaries[0])
CU1 = tensorP(unitaries[0], outerP(inputBasisVectors[0]))

for i in range(1, M):
    CU = CU + tensorP(outerP(inputBasisVectors[i]), unitaries[i])
    CU1 = CU1 + tensorP(unitaries[i], outerP(inputBasisVectors[i]))
    #print("A: ")
    #print(unitaries[i]*inputStates[i] - inputBasisVectors[i])

nonTargetOuterP = np.matrix(zeroMatrixArray(M, M))
for i in range(M):
    if (i!=1):
        nonTargetOuterP+=outerP(inputBasisVectors[i])
U = tensorP(identityMatrix(M*M), nonTargetOuterP) + tensorP(CU*SWAP, outerP(inputBasisVectors[1])) 
U1 = tensorP(identityMatrix(M*M), nonTargetOuterP) + tensorP(CU1*SWAP, outerP(inputBasisVectors[1]))

for i in range(Nq):
    rho = tensorP(rhoS, rhoC, rhoF)
    rho1 = tensorP(rhoC, rhoS, rhoF)
    transRho = U*rho*U.getH()
    transRho1 = U1*rho1*U1.getH()
    rhoSOut = partialTrace(transRho, M)
    rhoC = partialTrace(transRho1, M)
    #print(rhoSOut)
    epsilon = 0
    for j in range(M):
        if (j!=inputStateIndex):
            epsilon+=rhoSOut[j, j]

    plt.scatter(i+1, epsilon, s= 30, c= 'r', alpha = 0.5)
    #print(epsilon)
##x=np.arange(1.0, Nq, 0.02)
#plt.plot(x, 0.5*gamma + (1+gamma)**(x-1)*(1-gamma)/2**x, 'b--')


plt.xlabel("N")
plt.ylabel("Error Prob.")
plt.show()


