import numpy as np
import matplotlib.pyplot as plt
import math
import random

zero = np.matrix([[1], [0]])
one = np.matrix([[0], [1]])
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
        if abs(ket1[i, 0] - ket2[i, 0])>epsilon:
            result = False
            
    return result

def innerP(ket1, ket2):
    return trace(ket1.T*ket2)

def maxMixedState(dim):
    return identityMatrix(dim)/dim

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
            if(norm(vector)>0):
                vector = vector/norm(vector)
                '''the process works if a set of input states is chosen such that vector may turn out to be the zero vector. Example: linearly dependent non orthogonal states'''
            bmBasis.append(vector)
         
            Sum = np.matrix(zeroMatrixArray(M, 1))
            counter = 0
            
            for m in range(M):
                '''cycles through all vectors in set of input states and selects ones not used yet. A state may be spanned by bmBasis if its
                projection onto bmBasis is equal to itself.'''
                if (ketApproxEqual(projection(inputStates[m], bmBasis), inputStates[m], 10**(-8)) and not(m in usedStates)):
                    Sum = Sum + inputBasisVectors[m]
                    counter+=1

            #counter is guaranteed to be non-zero
            cmBasis.append(1/math.sqrt(counter)*Sum)
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
            quantity += inputProbWeights[j]*pMatrices[i][i, j] #finds coefficient of CTC coefficient
        if(quantity>Max):
            Max = quantity
            index = i
    CTCProbWeights = zeroMatrixArray(1, dim)
    CTCProbWeights[0][index]=1
    CTCProbWeights = np.matrix(CTCProbWeights)
    plotGraphAvgProbMarkov(N, inputBasisVectors, inputProbWeights, CTCProbWeights, inputStates, unitaries, 'r') #plots optimized curve in red
    plotGraphAvgProbMarkov(N, inputBasisVectors, inputProbWeights, maxMixedState, inputStates, unitaries, 'b') #plots curve with CTC initialized to maximally mixed state in blue

#Markov method applied for BB84 states
    
I=identityMatrix(2)
unitaries = [SWAP, tensorP(X,X), tensorP(X,I)*tensorP(HAD,I), tensorP(X,HAD)*SWAP]
inputBasisVectors = [tensorP(zero, zero),
                     tensorP(zero, one),
                     tensorP(one, zero),
                     tensorP(one, one)]
BB84InputStates = [tensorP(zero, zero), tensorP(one, zero), tensorP(plus, zero), tensorP(minus, zero)]
print(probTransMatrix(inputBasisVectors, BB84InputStates[0], unitaries))
CTCProbWeights = [.25, .25, .25, .25] #1/4*(|00> + |01> + |10> + |11>)
inputProbWeights = [.85, .05, .05, .05]
Nq = 100
#plotGraphAvgProbMarkov(Nq, inputBasisVectors, inputProbWeights, CTCProbWeights, BB84InputStates, unitaries, 'g')
optimizeCTC(Nq, inputBasisVectors, inputProbWeights, BB84InputStates, unitaries)

plt.show()









