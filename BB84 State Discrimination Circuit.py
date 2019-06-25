import numpy as np
import matplotlib.pyplot as plt
import math

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
    dimension=4
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

Nq = 100
gamma = 0
registerVector = np.matrix([[float(math.sqrt(gamma))], [float(math.sqrt(1-gamma))]])
rhoF = outerP(registerVector)
inputStates = [zero, one, plus, minus]
inputStatesProbWeights = [1,0,0,0]#[.25,.25,.25,.25]


indices0 = [0, 1, 2, 3]
indices1 = [2, 3, 0, 1]

#0. rho = rhoS1 * rhoS2 * rhoC1 * rhoC2 * rhoF

U0 = tensorP(identityMatrix(16), outerP(zero)) + tensorP(u11(indices0)*u10(indices0)*u01(indices0)*u00(indices0)*SWAPE(1,3)*SWAPE(0,2), outerP(one))

#1. rho = rhoC1 * rhoC2 * rhoS1 * rhoS2 * rhoF

U1 = tensorP(identityMatrix(16), outerP(zero)) + tensorP(u11(indices1)*u10(indices1)*u01(indices1)*u00(indices1)*SWAPE(1,3)*SWAPE(0,2), outerP(one))

pi = .5*outerP(zero) + .5*outerP(one)
#successProb = [zero, one, plus, minus]
successProb = [0, 0, 0, 0]
#rhoCList = [zero, one, plus, minus]
rhoCList = [tensorP(pi, pi), tensorP(pi, pi), tensorP(pi, pi), tensorP(pi, pi)]


for i in range(Nq):

    for j in range(4):
        inputIndex = j

        rhoS = tensorP(outerP(inputStates[inputIndex]), outerP(zero))
        rhoC = rhoCList[inputIndex]
        
        #0. rho = rhoS1 * rhoS2 * rhoC1 * rhoC2 * rhoF
        
        rho0 = tensorP(rhoS, rhoC, rhoF)

        #1. rho = rhoC1 * rhoC2 * rhoS1 * rhoS2 * rhoF
        
        rho1 = tensorP(rhoC, rhoS, rhoF)

            
        transRho0 = U0*rho0*U0.getH()
        transRho1 = U1*rho1*U1.getH()
        
        rhoSOut = partialTrace(transRho0)
        rhoCOut = partialTrace(transRho1)        
        
        if (inputIndex == 0):
            #zero
            s = rhoCOut[0,0]
            if (i==0):
                print(rhoCOut)
        elif (inputIndex == 1):
            #one
            s = rhoCOut[1,1]
        elif (inputIndex == 2):
            #plus
            s = rhoCOut[2,2]
        elif (inputIndex == 3):
            #minus
            s = rhoCOut[3,3]
        else:
            print("ERROR")
            
        #print(trace(rhoCOut))


        rhoCList[inputIndex] = rhoCOut
        successProb[inputIndex] = s
        

    averageSuccessProb = inputStatesProbWeights[0]*successProb[0] + inputStatesProbWeights[1]*successProb[1] + inputStatesProbWeights[2]*successProb[2] + inputStatesProbWeights[3]*successProb[3]
    N=i+1

    #print(averageSuccessProb)    
    
    plt.scatter(N, averageSuccessProb, s= 30, c= 'r', alpha = 0.5)

plt.xlabel("N")
plt.ylabel("Success Prob.")
plt.show()

'''commented out sections correspond to less ugly versions of some BB84 functions that I hard coded in a while ago. I have not tested them yet though.'''
##def SWAPmultiQubit(pos1, pos2, Nsystems):
##    tensorPTemplate = []
##    for i in range(Nsystems):
##        tensorPTemplate.append(identityMatrix(2))
##
##    returnSWAP = np.matrix(zeroMatrixArray(2**Nsystems, 2**Nsystems))
##
##    for i in range(1):
##        for j in range(1):
##            tensorPTemplate[pos1] = inputBasisVectors[i]*inputBasisVectors[j].getH()
##            tensorPTemplate[pos2] = inputBasisVectors[j]*inputBasisVectors[i].getH()
##
##            tensorProduct = tensorPTemplate[0]
##            for k in range(1, Nsystems):
##                tensorProduct = t2p(tensorProduct, tensorPTemplate[k])
##            returnSWAP += tensorProduct
##
##    return returnSWAP

##def u00(indexList):
##    inputBasisVectors = generateInputBasisVectors(2)
##    
##    S1 = indexList[0]
##    S2 = indexList[1]
##    C1 = indexList[2]
##    C2 = indexList[3]
##    
##    tensorPTemplate = [identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2)]
##    tensorPTemplate[S1] = outerP(inputBasisVectors[0])
##    tensorPTemplate[S2] = outerP(inputBasisVectors[0])
##
##    returnU00 = tensorP(tensorPTemplate[0], tensorPTemplate[1], SWAP)
##    
##    for i in range(2):
##        for j in range(2):
##            if (i!=0 and j!=0):
##                tensorPTemplate[S1] = outerP(inputBasisVectors[i])
##                tensorPTemplate[S2] = outerP(inputBasisVectors[j])
##                tensorPTemplate[C1] = identityMatrix(2)
##                tensorPTemplate[C2] = identityMatrix(2)
##                returnU00 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
##
##    return returnU00
##
##
##def u01(indexList):
##    inputBasisVectors = generateInputBasisVectors(2)
##    
##    S1 = indexList[0]
##    S2 = indexList[1]
##    C1 = indexList[2]
##    C2 = indexList[3]
##
##    tensorPTemplate = [identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2)]
##    tensorPTemplate[S1] = outerP(inputBasisVectors[0])
##    tensorPTemplate[S2] = outerP(inputBasisVectors[1])
##    tensorPTemplate[C1] = X
##    tensorPTemplate[C2] = X
##    returnU01 = tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
##
##    for i in range(2):
##        for j in range(2):
##            if (i!=0 and j!=1):
##                tensorPTemplate[S1] = outerP(inputBasisVectors[i])
##                tensorPTemplate[S2] = outerP(inputBasisVectors[j])
##                tensorPTemplate[C1] = identityMatrix(2)
##                tensorPTemplate[C2] = identityMatrix(2)
##                returnU01 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
##                
##    return returnU01
##
##def u10(indexList):
##    inputBasisVectors = generateInputBasisVectors(2)
##    
##    S1 = indexList[0]
##    S2 = indexList[1]
##    C1 = indexList[2]
##    C2 = indexList[3]
##
##    tensorPTemplate = [identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2)]
##    tensorPTemplate[S1] = outerP(inputBasisVectors[1])
##    tensorPTemplate[S2] = outerP(inputBasisVectors[0])
##    tensorPTemplate[C1] = X*HAD
##    tensorPTemplate[C2] = identityMatrix(2)*identityMatrix(2)
##    returnU10 = tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
##
##    for i in range(2):
##        for j in range(2):
##            if (i!=1 and j!=0):
##                tensorPTemplate[S1] = outerP(inputBasisVectors[i])
##                tensorPTemplate[S2] = outerP(inputBasisVectors[j])
##                tensorPTemplate[C1] = identityMatrix(2)
##                tensorPTemplate[C2] = identityMatrix(2)
##                returnU10 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
##                
##    return returnU10
##
##def u11(indexList):
##    inputBasisVectors = generateInputBasisVectors(2)
##    
##    S1 = indexList[0]
##    S2 = indexList[1]
##    C1 = indexList[2]
##    C2 = indexList[3]
##
##    tensorPTemplate = [identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2)]
##
##    tensorPTemplate[S1] = identityMatrix(2)
##    tensorPTemplate[S2] = identityMatrix(2)
##    tensorPTemplate[C1] = X
##    tensorPTemplate[C2] = HAD
##    returnU11 = tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
##
##    tensorPTemplate[S1] = outerP(one) 
##    tensorPTemplate[S2] = outerP(one)
##    returnU11SUB = tensorP(tensorPTemplate[0], tensorPTemplate[1], SWAP)
##
##    returnU11*=returnU11SUB
##
##    for i in range(2):
##        for j in range(2):
##            if (i!=1 and j!=1):
##                tensorPTemplate[S1] = outerP(inputBasisVectors[i])
##                tensorPTemplate[S2] = outerP(inputBasisVectors[j])
##                tensorPTemplate[C1] = identityMatrix(2)
##                tensorPTemplate[C2] = identityMatrix(2)
##                returnU11 += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3])
##    
##    return returnU11

