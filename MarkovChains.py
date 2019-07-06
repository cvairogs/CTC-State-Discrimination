import numpy as np
import matplotlib.pyplot as plt
import math
import random
import cmath

##zero = np.matrix([[1], [0]])
##one = np.matrix([[0], [1]])
zero = np.matrix([[1], [0], [0]])
one = np.matrix([[0], [1], [0]])
two = np.matrix([[0], [0], [1]])
plus = np.matrix([[1/math.sqrt(2)], [1/math.sqrt(2)]])
minus = np.matrix([[1/math.sqrt(2)], [-1/math.sqrt(2)]])
HAD = 1/math.sqrt(2)*np.matrix([[1,1],[1,-1]])
SWAP = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
X = np.matrix([[0,1],[1,0]])

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

#kroenecker product of two matrices
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

'''accepts argument in form of [matrix1, matrix2, ..., matrixN]
and gives tensor product of those states in that order'''
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

'''accepts matrix and returns matrix of elements ranging from rows
r1 to r2 and columns c1 to c2. Row and column numbering starts at 0.
Useful for taking partial trace.  '''
def excise(matrix, r1, r2, c1, c2):
    returnMatrixArray = zeroMatrixArray(r2-r1+1, c2-c1+1)
    for i in range(r2-r1+1):
        for j in range(c2-c1+1):
            returnMatrixArray[i][j] = matrix[r1+i, c1+j]
    returnMatrix = np.matrix(returnMatrixArray)
    return returnMatrix

def partialTrace(tensor, dimension): 
    tRows = tensor.shape[0]
    tCols = tensor.shape[1]
    returnMatrixArray = []

    '''breaks up matrix into MxM blocks and takes trace of each
    of those blocks'''
    for i in range(dimension):
        returnMatrixArray.append([])
        for j in range(dimension):
            returnMatrixArray[i].append(trace(excise(tensor, int(i*tRows/dimension), int((i+1)*tRows/dimension-1), int(j*tCols/dimension), int((j+1)*tCols/dimension-1))))
    returnMatrix = np.matrix(returnMatrixArray)
    return returnMatrix

'''projects vector onto given basis. Because all the vectors
in basis are represented in the computational basis along with the
vector ket, the result should also be a matrix representing them in
the computational basis. This means projection returns the same matrix
if the states in basis can span the space in which ket is contained in
(projection would not return the same matrix if ket had say an extra component)
'''
def projection(ket, basis):
    dimension = ket.shape[0]
    returnKet = np.matrix(zeroMatrixArray(dimension, 1))
    for i in range(len(basis)):
        returnKet = returnKet + outerP(basis[i])*ket
    return returnKet


'''checks that two vectors are equal with an error tolerance epsilon'''
def ketApproxEqual(ket1, ket2, epsilon):
    result = True
    for i in range(ket1.shape[0]):
##        print(ket1[i,0])
##        print(ket2[i,0])
        if abs(ket1[i, 0] - ket2[i, 0])>epsilon:
            result = False
            
    return result

def innerP(ket1, ket2):
    return trace(ket1.getH()*ket2)

def maxMixedState(dim):
    return identityMatrix(dim)/dim

def maxMixedArray(dim):
    returnArray = []
    for i in range(dim):
        returnArray.append(1/dim)
    return returnArray

def genRandVector(dim): #dim is dimension
    elements = []
    norm2 = 0 #the norm that all the not yet assigned components must add up to after each iteration
    
    for i in range(dim):
        elements.append([0])
    for i in range(dim):
        if(i<(dim-1)):
            multiplier = random.uniform(0,1) 
            elementNorm = random.uniform(0, math.sqrt(1-norm2)*multiplier)
            '''if there is no multiplier, then for vectors of large dimension, there will be many components with zero in them because
            the magnitudes of each element before the zeros are assigned too high, i.e. the norm of all components will add up to 1 too soon'''
            phase = random.uniform(0, 2*math.pi)
            elementReal = math.cos(phase)*elementNorm
            elementImag = math.sin(phase)*elementNorm*1j
            element = elementReal + elementImag
            elements[i]=[element]
            norm2+=abs(element)**2
            if abs(norm2-1)<10**(-5):
                break
        else: #last component is assigned so that norm is guarantee to be one
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
    ''' constructs basis vectors by taking slices of identity matrix, which will
    have columns with a single 1 along them'''
    inputBasisVectors = []
    for i in range(M):
        inputBasisVectors.append(excise(identityMatrix(M), 0, M-1, i, i))
    return inputBasisVectors

def generateUnitaries(inputBasisVectors, inputStates):
    M = len(inputBasisVectors)

    unitaries = []

    '''pretty much follows procedure as given in paper by Mark. constructs two sets of
    orthonormal bases for each unitary U_k. They are constructed so that
    inputBasisVectors[i] = generateUnitaries(.,.)[i]*inputStates[i]'''

    for k in range(M):
        cmBasis = []
        bmBasis = []
        usedStates = []

        #Step 1
        bmBasis.append(inputStates[k])
        usedStates.append(k)

        #Step 2
        cmBasis.append(inputBasisVectors[k])

        for l in range(M-1):
            ''' selects a random vector from the set of input states that has not been used yet.
            orthogonalizes it using states already in bmBasis.'''
            index = random.randint(0, len(inputStates)-1)
            while (index in usedStates):
                index = random.randint(0, len(inputStates)-1)
            outerPSum = np.matrix(zeroMatrixArray(M, M))
            for n in range(len(bmBasis)):
                outerPSum = outerPSum + outerP(bmBasis[n])
            vector = inputStates[index] - outerPSum*inputStates[index]
            trig = False
            if(norm(vector)>10**(-3)):
                vector = vector/norm(vector)
                '''the process works if a set of input states is chosen such that vector may turn out to be the zero vector. Example: linearly dependent non orthogonal states'''
                bmBasis.append(vector)
                trig = True
         
            Sum = np.matrix(zeroMatrixArray(M, 1))
            counter = 0
            #print(trig)
            
            for m in range(M):
                '''cycles through all vectors in set of input states and selects ones not used yet. A state may be spanned by bmBasis if its
                projection onto bmBasis is equal to itself.'''
                if (ketApproxEqual(projection(inputStates[m], bmBasis), inputStates[m], 10**(-7)) and not(m in usedStates)):
                    Sum = Sum + inputBasisVectors[m]
                    counter+=1

            #counter is guaranteed to be non-zero
            try:
                cmBasis.append(1/math.sqrt(counter)*Sum)
            except ZeroDivisionError:
                print("ZERO DIVISION ERROR")
                for i in range(len(bmBasis)):
                    for j in range(len(bmBasis)):
                        if (i!=j):
                            print(innerP(bmBasis[i], bmBasis[j]))
                print(inputStates)
                print(trig)
            usedStates.append(index)
            
        U=np.matrix(zeroMatrixArray(M, M))
        for j in range(M):
            U=U+cmBasis[j]*bmBasis[j].getH()
        unitaries.append(U)

    return unitaries

#matrix not in canonical form
def probTransMatrix(inputBasisVectors, inputState, unitaries):
    dim = len(inputBasisVectors)
    pMatrix = zeroMatrixArray(dim, dim)
    for i in range(dim):
        for j in range(dim):
            element = abs((inputBasisVectors[j].getH()*unitaries[i]*inputState)[0,0])**2
            pMatrix[i][j] = element
    return np.matrix(pMatrix)


def probTransMatrixN(N, inputBasisVectors, inputState, unitaries):
    pMatrix = probTransMatrix(inputBasisVectors, inputState, unitaries)
    return pMatrix**N

def QMatrix(inputBasisVectors, inputStateIndex, inputState, unitaries):
    dim = len(inputBasisVectors)
    qMatrix = zeroMatrixArray(dim-1, dim-1)
    k = inputStateIndex
    for i in range(dim-1):
        for j in range(dim-1):
            element = 0
            if (i<k and j<k):
                element = abs((inputBasisVectors[j].getH()*unitaries[i]*inputState)[0,0])**2
            if (i>=k and j<k):
                element = abs((inputBasisVectors[j].getH()*unitaries[i+1]*inputState)[0,0])**2
            if (i<k and j>=k):
                element = abs((inputBasisVectors[j+1].getH()*unitaries[i]*inputState)[0,0])**2
            if (i>=k and j>=k):
                element = abs((inputBasisVectors[j+1].getH()*unitaries[i+1]*inputState)[0,0])**2
            qMatrix[i][j] = element
    return np.matrix(qMatrix)

#plots average probability of success. All arguments besides color and Nq accepted as lists
def plotGraphAvgProbMarkov(Nq, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries, color):
    
    dim = len(inputBasisVectors)
    for i in range(Nq):
        averageProb=0
        for j in range(dim):
            inputStateIndex = j
            pN=probTransMatrixN(i+1, inputBasisVectors, inputStates[j], unitaries)
            uN= CTCProbWeights*pN #CTCProbWeights if denoted as matrix u in notes
            #print(p)
            Sum = 0 
            
            Sum+=uN[0, inputStateIndex]
            averageProb+=inputProbWeights[inputStateIndex]*Sum

        plt.scatter(i+1, averageProb, s=30, c=color, alpha = 0.5)
        
##    plt.show()

'''maximizes the probability of success after N iterations by setting the initial CTC state as described in notes'''
def optimizeCTC(N, inputBasisVectors, inputProbWeights, inputStates, unitaries):
    dim = len(inputBasisVectors)
    pMatrices = []

    maxMixedState = []

    for i in range(dim):
        maxMixedState.append(1/dim)
    
    for i in range(dim):
        pMatrices.append(probTransMatrixN(N, inputBasisVectors, inputStates[i], unitaries))
    Max = 0
    index = 0
    for i in range(dim): #loops through CTC coefficients. Process corresponds to (12) in the notes 
        quantity = 0
        for j in range(dim):
            quantity += inputProbWeights[j]*pMatrices[j][i, j] #finds coefficient of CTC coefficient
        if(quantity>Max):
            Max = quantity
            index = i
    CTCProbWeights = zeroMatrixArray(1, dim)
    CTCProbWeights[0][index]=1
    CTCProbWeights = np.matrix(CTCProbWeights)
    plotGraphAvgProbMarkov(N, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries, 'r') #plots optimized curve in red
    plotGraphAvgProbMarkov(N, inputBasisVectors, inputProbWeights, maxMixedState, inputStates, unitaries, 'b') #plots curve with CTC initialized to maximally mixed state in blue

def calculateAverageSuccessProb(N, CTCProbWeights, inputBasisVectors, inputPWeights, inputStates, unitaries):
    averageProb=0
    CTCProbWeights = np.matrix([CTCProbWeights])
    dim = len(inputBasisVectors)
    for j in range(dim):
        inputStateIndex = j
        pN=probTransMatrixN(N, inputBasisVectors, inputStates[j], unitaries)
##        print("prob trans matrix for " + str(inputStates[j]))
##        print(probTransMatrix(inputBasisVectors, inputStates[j], unitaries))
        uN= CTCProbWeights*pN #CTCProbWeights is denoted as matrix u in notes
        Sum = 0 
        
        Sum+=uN[0, inputStateIndex]
        averageProb+=inputPWeights[inputStateIndex]*Sum
    return averageProb


##Helper functions for optimization
##==========================================

##def calculateAverageErrorProb2D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates):
##    phi1 = phiVector[0][0]
##    phi2 = phiVector[1][0]
##    phi3 = phiVector[2][0]
##    phi4 = phiVector[3][0]
##
##    b10 = inputStates[0]
##    b20 = inputStates[1] - outerP(b10)*inputStates[1]
##    b20 /=norm(b20)
##
##    b11 = inputStates[1]
##    b21 = inputStates[0] - outerP(b11)*inputStates[0]
##    b21/=norm(b21)
##
##    U0 = cmath.exp(1j*phi1)*zero*b10.getH() + cmath.exp(1j*(phi2-phi1))*one*b20.getH() 
##    U1 = cmath.exp(1j*phi3)*one*b11.getH() - cmath.exp(1j*(phi4-phi3))*zero*b21.getH()
##
##    unitaries = [U0, U1]
##
##    prob = 1- calculateAverageSuccessProb(N, CTCProbWeights, inputBasisVectors, inputProbWeights, inputStates, unitaries)
##    return [prob, unitaries]

def calculateAverageErrorProb3D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates):
    phi1 = phiVector[0][0]
    phi2 = phiVector[1][0]
    phi3 = phiVector[2][0]
    phi4 = phiVector[3][0]
    phi5 = phiVector[4][0]
    phi6 = phiVector[5][0]
    phi7 = phiVector[6][0]
    phi8 = phiVector[7][0]
    phi9 = phiVector[8][0]
    phi10 = phiVector[9][0]
    phi11 = phiVector[10][0]
    phi12 = phiVector[11][0]
    phi13 = phiVector[12][0]
    phi14 = phiVector[13][0]
    phi15 = phiVector[14][0]
    

    b10 = inputStates[0]
    b20 = inputStates[1] - outerP(b10)*inputStates[1]
    b20 /=norm(b20)
    b30 = inputStates[2] - (outerP(b20)+outerP(b10))*inputStates[2]
    b30 /=norm(b30)

    b11 = inputStates[1]
    b21 = inputStates[0] - outerP(b11)*inputStates[0]
    b21/=norm(b21)
    b31 = inputStates[2] - (outerP(b11) + outerP(b21))*inputStates[2]
    b31 /= norm(b31)

    b12 = inputStates[2]
    b22 = inputStates[1] - outerP(b12)*inputStates[1]
    b22 /= norm(b22)
    b32 = inputStates[0] - (outerP(b12) + outerP(b22))*inputStates[0]
    b32 /= norm(b32)
    
    U0 = cmath.exp(1j*phi1)*zero*b10.getH() + 1/math.sqrt(2)*(cmath.exp(1j*(phi2))*one + cmath.exp(1j*phi4)*two)*b20.getH() + 1/math.sqrt(2)*(cmath.exp(1j*phi3)*one+cmath.exp(1j*phi5)*two)*b30.getH()
    U1 = cmath.exp(1j*phi6)*one*b11.getH() + 1/math.sqrt(2)*(cmath.exp(1j*phi7)*zero + cmath.exp(1j*phi9)*two)*b21.getH() + 1/math.sqrt(2)*(cmath.exp(1j*phi8)*zero+cmath.exp(1j*phi10)*two)*b31.getH()
    U2 = cmath.exp(1j*phi11)*two*b12.getH() + 1/math.sqrt(2)*(cmath.exp(1j*phi12)*zero + cmath.exp(1j*phi14)*one)*b22.getH() + 1/math.sqrt(2)*(cmath.exp(1j*phi13)*zero + cmath.exp(1j*phi15)*one)*b32.getH()
        
    unitaries = [U0, U1, U2]

    prob = 1-calculateAverageSuccessProb(N, CTCProbWeights, inputBasisVectors, inputProbWeights, inputStates, unitaries)
    return [prob, unitaries]

##def gradAverageErrorProb(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates):
##    dPhi = .01
##    grad = [[0],[0],[0],[0]]
##
##    p = calculateAverageErrorProb2D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)
##    for i in range(3):
##        phiVector[i][0] += dPhi
##        pNew = calculateAverageErrorProb2D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)
##        grad[i][0] = (pNew - p)/dPhi
##        phiVector[i][0] -= dPhi
##
##    return grad


##def optimizeUnitariesFixedCTC(N, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates):
##    dStep = 0.01
##    zeroVector = np.matrix([[0],[0],[0],[0]])
##
##    phiVector = [[random.uniform(0, 2*math.pi)],[random.uniform(0, 2*math.pi)], [random.uniform(0, 2*math.pi)],[random.uniform(0, 2*math.pi)]]
##    gradP = gradAverageErrorProb(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)
##
##    
##    while (not ketApproxEqual(np.matrix(gradP), zeroVector, 10**(-3))):
##        P = calculateAverageErrorProb2D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)
##        gradP = gradAverageErrorProb(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)
##        for i in range(3):
##            phiVector[i][0] -= dStep*gradP[i][0]
##            print("one iteration")

##def displayAllPs(N, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates):
##    dPhi = .1
##    phi1 = 0
##    phi2 = 0
##    phi3 = 0
##    phi4 = 0
##    for i in range(math.floor(2*math.pi/dPhi)):
##        phi1 = i*dPhi
##        for j in range(math.floor(2*math.pi/dPhi)):
##            phi2 = j*dPhi
##            for l in range(math.floor(2*math.pi/dPhi)):
##                phi3 = l*dPhi
##                for m in range(math.floor(2*math.pi/dPhi)):
##                    phi4 = m*dPhi
##
##                    phiVector = [[phi1],[phi2],[phi3],[phi4]]
##                    P=calculateAverageErrorProb2D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)[0]
##                    unitaries = calculateAverageErrorProb2D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)[1]
##                    U0 = unitaries[0]
##                    U1 = unitaries[1]
####                    print(P)
####                    print(U0)
####                    print(U1)
####                    print("trans matrix 1: ")
####                    print(probTransMatrix(inputBasisVectors, inputStates[0], unitaries))
####                    print("trans matrix 2: ")
####                    print(probTransMatrix(inputBasisVectors, inputStates[1], unitaries))
        

def gradAverageErrorProb3D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates):
    dPhi = 0.01
    grad = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]

    p = calculateAverageErrorProb3D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)[0]
    for i in range(15):
        if (i!=4 and i!=9 and i!=14):
            phiVector[i][0] += dPhi
            pNew = calculateAverageErrorProb3D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)[0]
            #print((pNew - p)/dPhi)
            grad[i][0] = (pNew - p)/dPhi
            phiVector[i][0] -= dPhi
        else:
            grad[i][0] = 0

    #print('gradient is ')
    return grad

def optimizeUnitariesFixedCTC3D(N, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates):
    dStep = .01
    zeroVector = np.matrix([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])

    phi1 = random.uniform(0, 2*math.pi)
    phi2 = random.uniform(0, 2*math.pi)
    phi3 = random.uniform(0, 2*math.pi)
    phi4 = random.uniform(0, 2*math.pi)
    phi5 = phi4 - phi2 + math.pi + phi3
    phi6 = random.uniform(0, 2*math.pi)
    phi7 = random.uniform(0, 2*math.pi)
    phi8 = random.uniform(0, 2*math.pi)
    phi9 = random.uniform(0, 2*math.pi)
    phi10 = phi9 - phi7 + math.pi + phi8
    phi11 = random.uniform(0, 2*math.pi)
    phi12 = random.uniform(0, 2*math.pi)
    phi13 = random.uniform(0, 2*math.pi)
    phi14 = random.uniform(0, 2*math.pi)
    phi15 = phi14 - phi12 + math.pi + phi13
    phiVector = [[phi1], [phi2], [phi3], [phi4], [phi5], [phi6], [phi7], [phi8], [phi9], [phi10], [phi11], [phi12], [phi13], [phi14], [phi15]]
    
    gradP = gradAverageErrorProb3D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)

    
    while (not ketApproxEqual(np.matrix(gradP), zeroVector, 10**(-3))):
        P = calculateAverageErrorProb3D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)[0]
        gradP = gradAverageErrorProb3D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)
        gradPCopy = np.matrix(gradP)
        
        gradPCopy /= norm(gradPCopy)
        for i in range(15):
            if (i!=4 and i!= 9 and i!= 14):
                phiVector[i][0] -= dStep*gradPCopy[i, 0]
            else:
                phiVector[i][0] = phiVector[i-1][0] - phiVector[i-3][0] + math.pi + phiVector[i-2][0]
    
        print("norm " + str(norm(np.matrix(gradP))))


    return calculateAverageErrorProb3D(N, phiVector, CTCProbWeights, inputProbWeights, inputBasisVectors, inputStates)[1]
            


           



#Markov method applied for BB84 states
    
##I=identityMatrix(2)
##unitaries = [SWAP, tensorP(X,X), tensorP(X,I)*tensorP(HAD,I), tensorP(X,HAD)*SWAP]
##inputBasisVectors = [tensorP(zero, zero),
##                     tensorP(zero, one),
##                     tensorP(one, zero),
##                     tensorP(one, one)]
##BB84InputStates = [tensorP(zero, zero), tensorP(one, zero), tensorP(plus, zero), tensorP(minus, zero)]
##print(probTransMatrix(inputBasisVectors, BB84InputStates[0], unitaries))
##CTCProbWeights = [.25, .25, .25, .25] #1/4*(|00> + |01> + |10> + |11>)
##inputProbWeights = [.85, .05, .05, .05]
##Nq = 100
###plotGraphAvgProbMarkov(Nq, inputBasisVectors, inputProbWeights, CTCProbWeights, BB84InputStates, unitaries, 'g')
##optimizeCTC(Nq, inputBasisVectors, inputProbWeights, BB84InputStates, unitaries)

dim = 25


##for i in range(500):
##    inputBasisVectors = generateInputBasisVectors(dim)
##    inputStates = generateInputStates(dim)
##    unitaries = generateUnitaries(inputBasisVectors, inputStates)
##    CTCProbWeights = [1,0]
##    inputProbWeights = [.5,.5]
##    Nq = 10
##
##    error = 1-calculateAverageSuccessProb(Nq, CTCProbWeights, inputBasisVectors, inputProbWeights, inputStates, unitaries)
##
##    fidelity = 0
##    counter=0
##    for i in range(dim):
##        for j in range(i, dim):
##            if(i!=j):
##                fidelity+=abs(innerP(inputStates[i], inputStates[j]))**2
##                counter+=1
##    fidelity/=counter
##    
##    plt.scatter(fidelity, error, s=30, c = 'g', alpha = 0.5)

for i in range(100000):
    inputBasisVectors = generateInputBasisVectors(dim)
    inputStates = generateInputStates(dim)
    unitaries = generateUnitaries(inputBasisVectors, inputStates)
    CTCProbWeights = maxMixedArray(dim)
    inputProbWeights = maxMixedArray(dim)
    Nq = 10

    inputStateIndex = 3
    Q = QMatrix(inputBasisVectors, inputStateIndex, inputStates[inputStateIndex], unitaries)
    #P = probTransMatrix(inputBasisVectors, inputStates[inputStateIndex], unitaries)
    e, v = np.linalg.eig(Q)
    #print(e)
    print(Q)
    print(e)
    if (len(e)<dim-1):
        print("error")
        print(Q)
        print(e)
        print("here")
        break
    
    triggered = False
    for i in range(len(e)-1):
        for j in range(i+1, len(e)):
            if abs(e[i]-e[j])<10**(-6):
                print("error")
                print(Q)
                print(e)
                triggered = True

    if (triggered == True):
        break
    

'''
unitaries1 = optimizeUnitariesFixedCTC3D(Nq, [1,0,0], inputProbWeights, inputBasisVectors, inputStates)
unitaries2 = optimizeUnitariesFixedCTC3D(Nq, [0,1,0], inputProbWeights, inputBasisVectors, inputStates)
unitaries3 = optimizeUnitariesFixedCTC3D(Nq, [0,0,1], inputProbWeights, inputBasisVectors, inputStates)

plotGraphAvgProbMarkov(Nq, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries, 'g')
plotGraphAvgProbMarkov(Nq, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries1, 'r')
plotGraphAvgProbMarkov(Nq, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries2, 'b')
plotGraphAvgProbMarkov(Nq, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries3, 'c')
'''


plt.show()









