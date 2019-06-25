import numpy as np
import matplotlib.pyplot as plt
import math
import random
import warnings

zero = np.matrix([[1], [0]])
one = np.matrix([[0], [1]])
plus = np.matrix([[1/math.sqrt(2)], [1/math.sqrt(2)]])
minus = np.matrix([[1/math.sqrt(2)], [-1/math.sqrt(2)]])
plusY = np.matrix([[1/math.sqrt(2)], [1j/math.sqrt(2)]])
minusY = np.matrix([[1/math.sqrt(2)], [-1j/math.sqrt(2)]])

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

def tensorPower(matrix, power):
    returnMatrix = tensorP(matrix, matrix)
    if (power==1):
        return matrix
    elif (power==2):
        return returnMatrix
    else:
        for i in range(2, power):
            returnMatrix = tensorP(returnMatrix, matrix)
        return returnMatrix

def SWAPE(pos1, pos2):
    tensorPTemplate = [identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2), identityMatrix(2)]
    tensorPTemplate[pos1] = zero*zero.getH()#inputBasisVectors[0]*inputBasisVectors[0].getH()
    tensorPTemplate[pos2] = zero*zero.getH()#inputBasisVectors[0]*inputBasisVectors[0].getH()
    returnSWAP = tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3], tensorPTemplate[4], tensorPTemplate[5])
    
    tensorPTemplate[pos1] = zero*one.getH()#inputBasisVectors[0]*inputBasisVectors[1].getH()
    tensorPTemplate[pos2] = one*zero.getH()#inputBasisVectors[1]*inputBasisVectors[0].getH()
    returnSWAP += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3], tensorPTemplate[4], tensorPTemplate[5])

    tensorPTemplate[pos1] = one*zero.getH()#inputBasisVectors[1]*inputBasisVectors[0].getH()
    tensorPTemplate[pos2] = zero*one.getH()#inputBasisVectors[0]*inputBasisVectors[1].getH()
    returnSWAP += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3], tensorPTemplate[4], tensorPTemplate[5])

    tensorPTemplate[pos1] = one*one.getH()#inputBasisVectors[1]*inputBasisVectors[1].getH()
    tensorPTemplate[pos2] = one*one.getH()#inputBasisVectors[1]*inputBasisVectors[1].getH()
    returnSWAP += tensorP(tensorPTemplate[0], tensorPTemplate[1], tensorPTemplate[2], tensorPTemplate[3], tensorPTemplate[4], tensorPTemplate[5])
    
    return returnSWAP    

M = 8 #dimension of vector space
usedM = 6 #of dimensions actually used

basisVectors = []
for i in range(M):
    basisVectors.append(excise(identityMatrix(M), 0, M-1, i, i))
 
inputBasisVectors = []
for i in range(usedM):
    inputBasisVectors.append(excise(identityMatrix(M), 0, M-1, i, i))
                        
zeroIn = tensorP(zero, zero, zero)
oneIn = tensorP(one, zero, zero)
plusIn = tensorP(plus, zero, zero)
minusIn = tensorP(minus, zero, zero)
plusYIn = tensorP(plusY, zero, zero)
minusYIn = tensorP(minusY, zero, zero)

inputStates = [zeroIn, oneIn, plusIn, 
               minusIn, plusYIn, minusYIn]

#construct controlled unitary for each input state.

unitaries = []

for k in range(usedM):
    cmBasis = []
    bmBasis = []
    usedStates = []
    
    #b1
    bmBasis.append(inputStates[k])
    usedStates.append(k)

    #c1
    cmBasis.append(inputBasisVectors[k])

    #procedure below outlined in paper "localized closed timelike curves can perfectly distinguish quantum states"
    for l in range(usedM-1):
        
        
        #pick one of the input states that has not already been used in the G-S procedure
        index = random.randint(0, len(inputStates)-1)
        while (index in usedStates):
            index = random.randint(0, len(inputStates)-1)
        
        #construct projector onto subspace formed by vectors that are already orthonormalized
        outerPSum = np.matrix(zeroMatrixArray(M, M))
        for n in range(len(bmBasis)):
            outerPSum = outerPSum + outerP(bmBasis[n])
            
        #subtract projection of non orthogonal state onto orthonormalized vectors, normalize, and add
        #to set of orthonormal vectors
        
        vector = inputStates[index] - outerPSum*inputStates[index]
        if(norm(vector) > 10**(-3)):
            vector = vector/norm(vector)
            bmBasis.append(vector)

##        print("bmBasis")
##        print(bmBasis)
##        print("input state used to construct last element of bmBasis")
##        print(inputStates[index])

        #loop through all input states. Check if the input state is in the subspace spanned by vectors 
        #already in bmBasis and has not already been used in the G-S procedure (excluding the very 
        #last vector used in the G-S procedure). If so, add it to Sum.
        Sum = np.matrix(zeroMatrixArray(M, 1))
        counter = 0
        for m in range(usedM):
##            print("testing if the following vector can be spanned by bmBasis: ")
##            print(inputStates[m])
##            print("projection")
##            print(projection(inputStates[m], bmBasis))
##            print(ketApproxEqual(projection(inputStates[m], bmBasis), inputStates[m], 10**(-3)))
##            print(not(m in usedStates))
            if (ketApproxEqual(projection(inputStates[m], bmBasis), inputStates[m], 10**(-3)) and not(m in usedStates)):
                #print("it can")
                Sum = Sum + inputBasisVectors[m]
                counter+=1
        
        #normalize the sum. Note that counter should not be equal to zero at this point because the state labeled
        #by the variable 'index' is not in the set of used states, and is guaranteed to be in the subspace spanned by
        #bmBasis because we just applied the G-S procedure to it.
        cmBasis.append(1/math.sqrt(counter)*Sum)
        usedStates.append(index)

    #As outlined in paper unitaries are constructed so that,
    #U[i]*inputStates[i]=inputBasisVectors[i]
    U=np.matrix(zeroMatrixArray(M, M))
    for j in range(len(bmBasis)):
        U=U+cmBasis[j]*bmBasis[j].getH()
    unitaries.append(U)

Nq = 1000
gamma = 0.5
registerVectorArray = [[math.sqrt(gamma)], [math.sqrt(1-gamma)]]
###fill registerVectorArray with zeros so that the register vector is of the same dimension as the vector space
##if (M>2):
##    for i in range(2, M):
##        registerVectorArray.append([0])
registerVector = np.matrix(registerVectorArray)

#construct SWAP gate
##SWAP = np.matrix(zeroMatrixArray(M*M, M*M))
##for i in range(M):
##    for j in range(M):
##        SWAP+=(tensorP(inputBasisVectors[i]*inputBasisVectors[j].getH(), inputBasisVectors[j]*inputBasisVectors[i].getH()))

inputStateIndex = 2
pi = .5*outerP(zero) + .5*outerP(one)
rhoF = outerP(registerVector)
rhoS = outerP(inputStates[inputStateIndex])
rhoC = tensorP(pi, pi, pi)

#construct controlled unitary for circuit    
CU = tensorP(outerP(basisVectors[0]), unitaries[0])
CU1 = tensorP(unitaries[0], outerP(basisVectors[0]))
for i in range(1, M):
    if (i<=usedM-1):
        CU = CU + tensorP(outerP(basisVectors[i]), unitaries[i])
        CU1 = CU1 + tensorP(unitaries[i], outerP(basisVectors[i]))
    else:
        CU += tensorP(outerP(basisVectors[i]), identityMatrix(8))
        CU1 += tensorP(identityMatrix(8), outerP(basisVectors[i]))

##nonTargetOuterP = np.matrix(zeroMatrixArray(M, M))
##for i in range(usedM):
##    if (i!=1):
##        nonTargetOuterP+=outerP(inputBasisVectors[i])

#wrong
#print(SWAPE(0,3).shape)

U =  tensorP(identityMatrix(2**6), outerP(zero)) + tensorP(CU*SWAPE(0,3)*SWAPE(1,4)*SWAPE(2,5), outerP(one)) 
#print('done')

U1 = tensorP(identityMatrix(2**6), outerP(zero)) + tensorP(CU1*SWAPE(0,3)*SWAPE(1,4)*SWAPE(2,5), outerP(one))

for i in range(Nq):
    rho = tensorP(rhoS, rhoC, rhoF)
    rho1 = tensorP(rhoC, rhoS, rhoF)
    transRho = U*rho*U.getH()
    transRho1 = U1*rho1*U1.getH()
    rhoSOut = partialTrace(transRho, M)
    rhoCOut = partialTrace(transRho1, M)
    rhoC = rhoCOut

    #print(rhoSOut)
    s = rhoC[inputStateIndex, inputStateIndex]
    print(rhoCOut)

    if(1-abs(s))<.001:
        print(i)
        break
    
    plt.scatter(i+1, math.log(1-abs(s)), s= 30, c= 'r', alpha = 0.5)
    #print(epsilon)
##x=np.arange(1.0, Nq, 0.02)
#plt.plot(x, 0.5*gamma + (1+gamma)**(x-1)*(1-gamma)/2**x, 'b--')




plt.xlabel("N")
plt.ylabel("Success Prob.")
plt.show()


