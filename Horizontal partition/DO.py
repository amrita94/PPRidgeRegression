import numpy as np
from util import space_mapping
from operator import mul


#Data Owner
class DOwner:
    def __init__(self, X, Y, magnitude):
        self.X = X
        self.Y = Y
        self.magnitude = magnitude
        self.pk = None
        self.enc_A = self.enc_b = None        # compute Enc(A_k), Enc(b_k) locally


    def set_PK(self, pk):
        self.pk = pk


    def computeEnc_Ab(self):
        """
        Protocol-Horizontal Step2(local computation)

        :param X, Y: training dataset. X = (n,d) matrix, Y = n vector
        :param magnitude: parameter for data representation(if we use at most "l" fractional digits, then magnitude = 10**l)
        :return enc_A: encrypted trans(X)*X 
        :return enc_b: encrypted trans(X)*Y 
        """
        # convert a data representation domain(real -> Integer)
        # R => Z<l+r> => Z<N>
        X = (self.X * self.magnitude).astype(int)
        Y = (self.Y * self.magnitude).astype(int).tolist()
        X_trans = np.transpose(X).tolist()
        num_instances = self.X.shape[0]
        num_features = self.X.shape[1]

        # compute A = trans(X)*X, encrypt A
        self.enc_A = []
        for index in range(num_features):
            # To improve efficiency, compute a triangular matrix of A
            # Because A_ij = A_ji
            self.enc_A.append([0]*index + [self.pk.encrypt(x) for x in [space_mapping(sum(map(mul, X_trans[index], X_trans[j])), self.pk.n) for j in range(index, num_features)]])

        # compute b = trans(X)*Y, encrypt b
        self.enc_b = [self.pk.encrypt(b) for b in [space_mapping(sum(map(mul, X_trans[i], Y)), self.pk.n) for i in range(num_features)]]


    # To send enc_A and enc_b to MLE for merging datasets
    def getEnc_Ab(self):
        return self.enc_A, self.enc_b
