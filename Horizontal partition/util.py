import numpy as np
from copy import deepcopy
import gmpy2


# map into Zn ring space
def space_mapping(value, plain_space):
    """
    Map the value into a defined domain. (modulo reduction)
    we define that Z(N) is a integer domain where [-N/2, N/2]

    :param value: input value
    :param public_key_N: N of the public key. 
    :return value: mapped value
    """
    value %= plain_space
    if value*2 > plain_space:
        value -= plain_space
    return value


def gaussElimination(matrix, public_N):
    """
    Gaussian Elimination algorithm to solve linear equations.
      - step1 rearrange rows in matrix
      - step2 convert it into an upper triangular matrix
      - solve equations for the triangular matrix from bottom to top
    reference code(https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/)

    :param matrix: input (d, d+1) matrix // if we compute w where Aw=b, "matrix" is an augmented matrix (A|b)
    :param public_N: N of the public key. all arithmetic computation should work in Z(n) finite field.
    :return x: computed coefficients for linear equations
    """
    A = deepcopy(matrix)
    n = len(A)

    for i in range(n):
        # Search for maximum in this column
        maxEl = abs(A[i][i])
        maxRow = i
        for k in range(i+1, n):
            if abs(A[k][i]) > maxEl:
                maxEl = abs(A[k][i])
                maxRow = k

        # Swap maximum row with current row (column by column)
        tmp = A[maxRow][i:]
        A[maxRow][i:] = A[i][i:]
        A[i][i:] = tmp

        # Make all rows below this one 0 in current column
        #assert A[i][i] != 0, "A[i][i]==0, in i="+str(i)
        inv_A_ii = invert(A[i][i], public_N)
        for k in range(i+1, n):
            #c = A[k][i]/A[i][i]
            c =  space_mapping(A[k][i] * inv_A_ii, public_N)
            for j in range(i, n+1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] = space_mapping(A[k][j] - c * A[i][j], public_N)

    # Solve equation Ax=b for an upper triangular matrix A
    x = [0] * n
    for i in range(n-1, -1, -1):
        #x[i] = A[i][n]/A[i][i]
        assert A[i][i] != 0
        x[i] = space_mapping( A[i][n] * invert(A[i][i], public_N) , public_N)
        for k in range(i-1, -1, -1):
            A[k][n] =  space_mapping(A[k][n] - A[k][i] * x[i], public_N)
                
    return x


def compute_det(matrix, public_N):
    """
    Compute determinant of a matrix
      - step1 convert a matrix into an upper triangular matrix, partially using gaussian elimination method.
      - step2 product all diagonal elements

    :param matrix: input matrix(square)
    :param public_N: N of the public key. all arithmetic computation should work in Z(n) finite field.
    :return det_A: determinant of matrix
    """
    A = deepcopy(matrix)
    n = len(A)

    # convert into a triangular matrix (Gaussian Elimination)
    for i in range(n):
        assert A[i][i] != 0
        # Make all rows below this one 0 in current column
        inv_A_ii = invert(A[i][i], public_N)
        for k in range(i+1, n):
            #c = A[k][i]/A[i][i]
            c =  space_mapping(A[k][i] * inv_A_ii, public_N)
            for j in range(i, n):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] = space_mapping(A[k][j] - c * A[i][j], public_N)

    det_A = 1
    for i in range(n):
        det_A = space_mapping(det_A * A[i][i], public_N)
    
    return det_A


# Functions of Strassen algorithm for a matrix multiplication
  # Strassen algorithm description(https://en.wikipedia.org/wiki/Strassen_algorithm)
  # reference code(https://stackoverflow.com/questions/12867099/strassen-matrix-multiplication-close-but-still-with-bugs)
def add_m(a, b, pkn):
    """
    add "a" and "b". Both of them are scalars(integers) or vectors(lists).
    compute the sum of "a" and "b", element-wise. Returns a scalar if both "a" and "b" are scalars.

    :param a, b: input values
    :param public_N: N of the public key. all arithmetic computation should work in Z(n) finite field.
    :return d: a + b
    """
    if type(a) == list:
        num_dimension = len(a)
        d = []
        for i in range(num_dimension):
            c = []
            for j in range(num_dimension):
                temp = a[i][j] + b[i][j]
                if isinstance(temp, int):
                    c.append(space_mapping(temp, pkn))                    
                else:
                    c.append(temp)
            d.append(c)

    else:
        if isinstance(a, int) and isinstance(b, int):
            d = space_mapping(a + b, pkn)
        else:
            d = a + b
    return d


def sub_m(a, b, pkn):
    """
    subtract "a" and "b". Both of them are scalars(integers) or vectors(lists).
    compute the difference of "a" and "b", element-wise. Returns a scalar if both "a" and "b" are scalars.

    :param a, b: input values
    :param pkn: N of the public key. all arithmetic computation should work in Z(n) finite field.
    :return d: a - b
    """
    if type(a) == list:
        num_dimension = len(a)
        d = []
        for i in range(num_dimension):
            c = []
            for j in range(num_dimension):
                temp = a[i][j] - b[i][j]
                if isinstance(temp, int):
                    c.append(space_mapping(temp, pkn))                    
                else:
                    c.append(temp)
            d.append(c)
    else:
        if isinstance(a, int) and isinstance(b, int):
            d = space_mapping(a - b, pkn)
        else:
            d = a - b
    return d


def ijk(a, b, pkn): # multiply the two matrices
    """
    multiply the two matrices with a pure iterative algorithm.

    :param a, b: input matrices
    :param pkn: N of the public key. all arithmetic computation should work in Z(n) finite field.
    :return C: a*b
    """
    num_dimension = len(a)
    C = np.zeros((num_dimension, num_dimension), dtype='int').tolist()
    for i in range(num_dimension):
        for j in range(num_dimension):
            for k in range(num_dimension):
                temp = C[i][j] + a[i][k]*b[k][j]
                if isinstance(temp, int):
                    C[i][j] = space_mapping(temp, pkn)
                else:
                    C[i][j] = temp
    return C


'''
# Functions of Strassen algorithm for a matrix multiplication
  # Strassen algorithm description(https://en.wikipedia.org/wiki/Strassen_algorithm)
  # reference code(https://stackoverflow.com/questions/12867099/strassen-matrix-multiplication-close-but-still-with-bugs)
'''

def split(matrix): # split matrix into quarters 
    """
    partition a matrix into equally sized block matrices

    :param matrix: input matrix
    :return a: left-upper submatrix of input
    :return b: right-upper submatrix of input
    :return c: left-bottom submatrix of input
    :return d: right-bottom submatrix of input
    """
    n = len(matrix)
    if n % 2 == 0:
        half_n = n//2
        a = [matrix[i][:half_n] for i in range(half_n)]
        b = [matrix[i][half_n:] for i in range(half_n)]
        c = [matrix[i][:half_n] for i in range(half_n, n)]
        d = [matrix[i][half_n:] for i in range(half_n, n)]
    else:
        # zero-padding, if the length of matrix is odd.
        half_n = n//2 + 1
        a = [matrix[i][:half_n] for i in range(half_n)]
        b = [matrix[i][half_n:] + [0] for i in range(half_n)] 
        c = [matrix[i][:half_n] for i in range(half_n, n)] + [[0]*half_n]
        d = [matrix[i][half_n:] + [0] for i in range(half_n, n)] + [[0]*half_n]
    
    return a,b,c,d


def strassenR(a, b, q, pkn):
    """
    Main function of strassen algorithm. To multiply a and b, it works recursively

    :param a, b: input matrices
    :param q: length of imput matrices
    :param pkn: N of the public key. all arithmetic computation should work in Z(n) finite field.
    :return : a result matrix of multiplying "a" and "b"
    """
    if q == 1:    # base case: 1x1 matrix
        return [[a[0][0] * b[0][0]]]
    elif q in [3, 5, 9]: # applying heuristic observation to improve performance
        return ijk(a, b, pkn)
    else:
        #split matrices into quarters
        a11, a12, a21, a22 = split(a)
        b11, b12, b21, b22 = split(b)
        q_2 = len(a11)

        # p1 = (a11+a22) * (b11+b22)
        p1 = strassenR(add_m(a11,a22, pkn), add_m(b11,b22, pkn), q_2, pkn)
        # p2 = (a21+a22) * b11
        p2 = strassenR(add_m(a21,a22, pkn), b11, q_2, pkn)
        # p3 = a11 * (b12-b22)
        p3 = strassenR(a11, sub_m(b12,b22, pkn), q_2, pkn)
        # p4 = a22 * (b21-b11)
        p4 = strassenR(a22, sub_m(b21,b11, pkn), q_2, pkn)
        # p5 = (a11+a12) * b22
        p5 = strassenR(add_m(a11,a12, pkn), b22, q_2, pkn)
        # p6 = (a21-a11) * (b11+b12)
        p6 = strassenR(sub_m(a21,a11, pkn), add_m(b11,b12, pkn), q_2, pkn)
        # p7 = (a12-a22) * (b21+b22)
        p7 = strassenR(sub_m(a12,a22, pkn), add_m(b21,b22, pkn), q_2, pkn)

        # c11 = p1 + p4 - p5 + p7
        c11 = add_m(sub_m(add_m(p1, p4, pkn), p5, pkn), p7, pkn)
        # c12 = p3 + p5
        c12 = add_m(p3, p5, pkn)
        # c21 = p2 + p4
        c21 = add_m(p2, p4, pkn)
        # c22 = p1 + p3 - p2 + p6
        c22 = add_m(sub_m(add_m(p1, p3, pkn), p2, pkn), p6, pkn)

        #merge 4 sub-lists
        if q % 2 == 0:
            return [item[0] + item[1] for item in zip(c11+c21, c12+c22)]
        else:
            return [item[0] + item[1][:-1] for item in zip(c11+c21[:-1], c12+c22[:-1])]


# Lagrange-Gauss lattice basis reduction(for Ratioanl Reconstruction)
def latticeReduction(u, v):
    #q = round((u[0]*v[0] + u[1]*v[1])/(v[0]**2 + v[1]**2)) # round( dot(u,v)/(||v||^2) )
    q = (u[0]*v[0] + u[1]*v[1])//(v[0]**2 + v[1]**2) # round( dot(u,v)/(||v||^2) )
    r = [u_i - (q*v_i) for u_i, v_i in zip(u, v)]
    u = v
    v = r
    
    while(u[0]**2 + u[1]**2) > (v[0]**2 + v[1]**2):
        q = (u[0]*v[0] + u[1]*v[1])//(v[0]**2 + v[1]**2) # round( dot(u,v)/(||v||^2) )
        r = [u_i - (q*v_i) for u_i, v_i in zip(u, v)]
        u = v
        v = r

    return u, v


# Ratioanl reconstruction
def rationalRecon(vector, pkn):
    conv_vector = []
    for item in vector:            
        if pkn**2 < (item**2 + 1): #if ||u|| < ||v||, swap u and v
            u = [item, 1]
            v = [pkn, 0]
        else:             #||u||>||v||              
            u = [pkn, 0]
            v = [item, 1]
        
        result, _ = latticeReduction(u, v)
        conv_vector.append(result[0]/result[1])
        
    return conv_vector


def invert(a, b):
    """
    return int: x, where a * x == 1 mod b
    """
    return int(gmpy2.invert(a, b))


def transpose(matrix):
    return list(map(list, zip(*matrix)))
