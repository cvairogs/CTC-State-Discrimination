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
    factor2 = ket.getH()
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
        norm2+=abs(ket[i,0])**2
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

def innerP(ket1, ket2):
    return trace(ket1.T*ket2)

'this function does not really generate random vectors. good enough for test case though'
def genRandVector(dim):
    elements = []
    norm2 = 0
    for i in range(dim):
        elements.append([0])
    for i in range(dim):
        if(i<(dim-1)):
            multiplier = random.uniform(0,1)
            elementNorm = random.uniform(0, math.sqrt(1-norm2)*multiplier)
            phase = random.uniform(0, 2*math.pi)
            elementReal = math.cos(phase)*elementNorm
            elementImag = math.sin(phase)*elementNorm*1j
            element = elementReal + elementImag
            elements[i]=[element]
            norm2+=abs(element)**2
            if abs(norm2-1)<.005:
                break
        else:
            elementNorm = math.sqrt(1-norm2)
            phase = random.uniform(0, 2*math.pi)
            elementReal = math.cos(phase)*elementNorm
            elementImag = math.sin(phase)*elementNorm*1j
            element = elementReal + elementImag
            elements[dim-1] = [element]   
    
    return np.matrix(elements)

def generateInputStates(N):
    inputStates = []
    for i in range(N):
        inputStates.append(genRandVector(N))
    return inputStates

def generateInputBasisVectors(M):
    inputBasisVectors = []
    for i in range(M):
        inputBasisVectors.append(excise(identityMatrix(M), 0, M-1, i, i))
    return inputBasisVectors

def generateUnitaries(inputBasisVectors, inputStates):
    M = len(inputBasisVectors)

    unitaries = []

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
            normV = norm(vector)
            vector = vector/norm(vector)
            bmBasis.append(vector)
         
            Sum = np.matrix(zeroMatrixArray(M, 1))
            counter = 0
##            print("==================================================")
##            print("workin on " + str(l+2) + "th basis vector of cmBasis" + " and unitary " + str(k))
##            print("bmBasis")
##            print(bmBasis)
##            print("projection of input state onto bmBasis")
##            print(projection(inputStates[index], bmBasis))
##            print("input state")
##            print(inputStates[index])
##            print("input state reconstructed from bmBasis")
##            print(normV*bmBasis[len(bmBasis)-1] + outerPSum *inputStates[index])
##            for i in range(len(bmBasis)):
##                for j in range(len(bmBasis)):
##                    if(i!=j):
##                        print("inner p between basis vectors " + str(i) + " " + str(j))
##                        print(innerP(bmBasis[i], bmBasis[j]))
##            print("++++++++++++++++++++++++++++++++++++++++++++++++++")
            #print(bmBasis)
            for m in range(M):

    ##            triggered = False
    ##            print("input State")
    ##            print(inputStates[m])
    ##            print("projections equal?")
    ##            print(ketApproxEqual(projection(inputStates[m], bmBasis), inputStates[m], 10**(-8)))
                #print(not(m in usedStates))
                if (ketApproxEqual(projection(inputStates[m], bmBasis), inputStates[m], 10**(-8)) and not(m in usedStates)):
                    Sum = Sum + inputBasisVectors[m]
                    #triggered = True
                    counter+=1
                #print(triggered)
                    
            cmBasis.append(1/math.sqrt(counter)*Sum)
            usedStates.append(index)
    ##        print("==================================================")
            

    ##    print(k)
    ##    print(cmBasis)
    ##    print(bmBasis)

        U=np.matrix(zeroMatrixArray(M, M))
        for j in range(M):
            U=U+cmBasis[j]*bmBasis[j].getH()
        unitaries.append(U)

    return unitaries

def plotGraph(Nq, gamma, inputBasisVectors, inputStates, inputStateIndex, unitaries, saveInfo):
    M=len(inputBasisVectors)

    registerVectorArray = [[math.sqrt(gamma)], [math.sqrt(1-gamma)]]
    if (M>2):
        for i in range(2, M):
            registerVectorArray.append([0])
    registerVector = np.matrix(registerVectorArray)

    #construct SWAP gate
    SWAP = np.matrix(zeroMatrixArray(M*M, M*M))
    for i in range(M):
        for j in range(M):
            SWAP+=(tensorP(inputBasisVectors[i]*inputBasisVectors[j].getH(), inputBasisVectors[j]*inputBasisVectors[i].getH()))

    pi = identityMatrix(M)/M
    rhoF = outerP(registerVector)
    rhoS = outerP(inputStates[inputStateIndex])
    rhoC = pi

    CU = tensorP(outerP(inputBasisVectors[0]), unitaries[0])
    CU1 = tensorP(unitaries[0], outerP(inputBasisVectors[0]))

    for i in range(1, M):
        CU = CU + tensorP(outerP(inputBasisVectors[i]), unitaries[i])
        CU1 = CU1 + tensorP(unitaries[i], outerP(inputBasisVectors[i]))

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

        plt.scatter(i+1, math.log(abs(epsilon)), s=30, c= 'r', alpha = 0.5)

    plt.xlabel("N")
    plt.ylabel("Error Prob.")
    
    if(saveInfo[0] == True):
        plt.savefig(saveInfo[1])
    else:
        plt.show()

def iterNeeded(errorP, gamma, inputBasisVectors, inputStates, inputStateIndex, unitaries):
    M=len(inputBasisVectors)

    registerVectorArray = [[math.sqrt(gamma)], [math.sqrt(1-gamma)]]
    if (M>2):
        for i in range(2, M):
            registerVectorArray.append([0])
    registerVector = np.matrix(registerVectorArray)

    #construct SWAP gate
    SWAP = np.matrix(zeroMatrixArray(M*M, M*M))
    for i in range(M):
        for j in range(M):
            SWAP+=(tensorP(inputBasisVectors[i]*inputBasisVectors[j].getH(), inputBasisVectors[j]*inputBasisVectors[i].getH()))

    pi = identityMatrix(M)/M
    rhoF = outerP(registerVector)
    rhoS = outerP(inputStates[inputStateIndex])
    rhoC = pi

    CU = tensorP(outerP(inputBasisVectors[0]), unitaries[0])
    CU1 = tensorP(unitaries[0], outerP(inputBasisVectors[0]))

    for i in range(1, M):
        CU = CU + tensorP(outerP(inputBasisVectors[i]), unitaries[i])
        CU1 = CU1 + tensorP(unitaries[i], outerP(inputBasisVectors[i]))

    nonTargetOuterP = np.matrix(zeroMatrixArray(M, M))
    for i in range(M):
        if (i!=1):
            nonTargetOuterP+=outerP(inputBasisVectors[i])
    U = tensorP(identityMatrix(M*M), nonTargetOuterP) + tensorP(CU*SWAP, outerP(inputBasisVectors[1])) 
    U1 = tensorP(identityMatrix(M*M), nonTargetOuterP) + tensorP(CU1*SWAP, outerP(inputBasisVectors[1]))

    epsilon = 10**100
    counter = 0
    while(epsilon>errorP/100):
        
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
        print(epsilon)
        counter+=1
    return counter

def plotGraphAvgProb(Nq, gamma, inputProbWeights, inputBasisVectors, inputStates, unitaries, color, saveInfo):
    M=len(inputBasisVectors)
    rhoCList = []
    pi = identityMatrix(M)/M#outerP(inputBasisVectors[2])#identityMatrix(M)/M

    for i in range(len(inputStates)):
        rhoCList.append(pi)
    
    registerVectorArray = [[math.sqrt(gamma)], [math.sqrt(1-gamma)]]
    if (M>2):
        for i in range(2, M):
            registerVectorArray.append([0])
    registerVector = np.matrix(registerVectorArray)

    #construct SWAP gate
    SWAP = np.matrix(zeroMatrixArray(M*M, M*M))
    for i in range(M):
        for j in range(M):
            SWAP+=(tensorP(inputBasisVectors[i]*inputBasisVectors[j].getH(), inputBasisVectors[j]*inputBasisVectors[i].getH()))

    CU = tensorP(outerP(inputBasisVectors[0]), unitaries[0])
    CU1 = tensorP(unitaries[0], outerP(inputBasisVectors[0]))

    for i in range(1, M):
        CU = CU + tensorP(outerP(inputBasisVectors[i]), unitaries[i])
        CU1 = CU1 + tensorP(unitaries[i], outerP(inputBasisVectors[i]))

    nonTargetOuterP = np.matrix(zeroMatrixArray(M, M))
    for i in range(M):
        if (i!=1):
            nonTargetOuterP+=outerP(inputBasisVectors[i])
    U = tensorP(identityMatrix(M*M), nonTargetOuterP) + tensorP(CU*SWAP, outerP(inputBasisVectors[1])) 
    U1 = tensorP(identityMatrix(M*M), nonTargetOuterP) + tensorP(CU1*SWAP, outerP(inputBasisVectors[1]))

    for i in range(Nq):
        averageError = 0
        for j in range(len(inputStates)):
            inputStateIndex = j           

            rhoF = outerP(registerVector)
            rhoS = outerP(inputStates[inputStateIndex])
            rhoC = rhoCList[inputStateIndex]
            
            rho = tensorP(rhoS, rhoC, rhoF)
            rho1 = tensorP(rhoC, rhoS, rhoF)
            transRho = U*rho*U.getH()
            transRho1 = U1*rho1*U1.getH()
            rhoSOut = partialTrace(transRho, M)
            rhoCOut = partialTrace(transRho1, M)
            rhoCList[inputStateIndex] = rhoCOut
            #print(rhoS*rhoS)
            #print(rhoSOut)
            
            errorProb = rhoCOut[inputStateIndex, inputStateIndex]
            averageError += errorProb*inputProbWeights[inputStateIndex]
        #print(averageError)
        plt.scatter(i+1, abs(averageError), s=30, c = color, alpha = 0.5)

    plt.xlabel("N")
    plt.ylabel("Success Prob.")
    
##    if(saveInfo[0] == True):
##        plt.savefig(saveInfo[1])
##    else:
##        plt.show()

def probTransMatrix(inputBasisVectors, inputState, unitaries):
    dim = len(inputBasisVectors)
    pMatrix = zeroMatrixArray(dim, dim)
    for i in range(dim):
        for j in range(dim):
            element = abs((inputBasisVectors[j].getH()*unitaries[i]*inputState)[0,0])**2
            pMatrix[i][j] = element
    return np.matrix(pMatrix)

##def canonicalProbTransMatrix(inputBasisVectors, inputState, inputStateIndex, unitaries):
##    dim = len(inputBasisVectors)
##    pMatrix = zeroMatrixArray(dim, dim)
##    for i in range(dim-1):
##        for j in range(dim-1):
##            element = abs((inputBasisVectors[j].getH()*unitaries[i]

def probTransMatrixN(N, inputBasisVectors, inputState, unitaries):
    pMatrix = probTransMatrix(inputBasisVectors, inputState, unitaries)
    return pMatrix**N

def plotGraphAvgProbMarkov(Nq, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries, color):
    
    dim = len(inputBasisVectors)
    for i in range(Nq):
        averageProb=0
        for j in range(dim):
            inputStateIndex = j
            pN=probTransMatrixN(i+1, inputBasisVectors, inputStates[j], unitaries)
            p= CTCProbWeights*pN
            #print(p)
            Sum = 0
            
            Sum+=p[0, inputStateIndex]
            averageProb+=inputProbWeights[inputStateIndex]*Sum

        plt.scatter(i+1, averageProb, s=30, c=color, alpha = 0.5)
        
##    plt.show()

def optimizeCTC(N, inputBasisVectors, inputProbWeights, inputStates, unitaries):
    dim = len(inputBasisVectors)
    pMatrices = []
    for i in range(dim):
        pMatrices.append(probTransMatrixN(N, inputBasisVectors, inputStates[i], unitaries))
    Max = 0
    index = 0
    for i in range(dim): #loops through CTC coefficients
        quantity = 0
        for j in range(dim):
            quantity += inputProbWeights[j]*pMatrices[i][i, j] #finds coefficient of CTC coefficient
        if(quantity>Max):
            Max = quantity
            index = i
    CTCProbWeights = zeroMatrixArray(1, dim)
    CTCProbWeights[0][index]=1
    CTCProbWeights = np.matrix(CTCProbWeights)
    plotGraphAvgProbMarkov(N, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries, 'r')
    plotGraphAvgProbMarkov(N, inputBasisVectors, inputProbWeights, np.matrix([[1/3.0,1/3.0,1/3.0]]), inputStates, unitaries, 'b')
    
def plotUpperBound(Nq, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries, color):
    dim = len(inputBasisVectors)
    pFailureList = []
    for i in range(dim):
        maxPFailure = 0
        for j in range(dim):
            test=1-probTransMatrix(inputBasisVectors, inputStates[i], unitaries)[j, i]
            print(test)
            if test>maxPFailure:
                maxPFailure = test
        pFailureList.append(maxPFailure)

    for i in range(Nq):
        avgError=0
        for j in range(dim):
            inputStateIndex = j
            avgError += (dim-1)*pFailureList[j]**(i+1)*inputProbWeights[j]
            avgError = min(1, avgError)
        plt.scatter(i+1, 1-avgError, s=30, c=color, alpha = 0.5)




Nq = 100
gamma = 0
dim = 2
#6inputStateIndex = 1
inputBasisVectors = generateInputBasisVectors(dim)
inputStates = generateInputStates(dim)
#print(inputStates)
unitaries = generateUnitaries(inputBasisVectors, inputStates)
inputProbWeights = [.5,.5]
CTCProbWeights = np.matrix([[.5,.5]])
saveInfo = [False, "NA"]

##for i in range(len(unitaries)):
##    print(unitaries[i]*inputStates[i])



#plots average probability of success as given by the trace out procedure and Markov chains. shows exact agreement.
plotGraphAvgProb(Nq, gamma, inputProbWeights, inputBasisVectors, inputStates, unitaries, 'g', saveInfo)
plotGraphAvgProbMarkov(Nq, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries, 'r')



##print(probTransMatrix(inputBasisVectors, inputStates[2], unitaries))
#print(probTransMatrix(inputBasisVectors, inputStates[2], unitaries))
#print(probTransMatrixN(80, inputBasisVectors, inputStates[2], unitaries))
#optimizeCTC(15, inputBasisVectors, inputProbWeights, inputStates, unitaries)
#plotUpperBound(Nq, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries, 'b')
plt.show()





