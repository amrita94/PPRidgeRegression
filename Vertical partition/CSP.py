import numpy as np
import operator
from labScheme import labGen, PRF
from util import space_mapping, gaussElimination, transpose
from copy import deepcopy


#Crypto Service Provider
class CSProvider:
    def __init__(self):
        self.mpk = self.msk = None
        self.list_mask_X = [] # PRF(Dec(upk), label)
        self.mask_Y = None


    # Protocol-Vertical Step1(key generation)
    def keyGen(self, n_length):
        self.mpk, self.msk = labGen(n_length)


    def compute_merged_mask(self, enc_seed_i, num_instances, num_features):
        """
        Protocol-Vertical Step1(set up)
        - receive upk_i[Enc(seed_i)], label information[num_instances, num_features] from DataOweners
        - compute B'=Enc(sum(b_i*b_j), c'=Enc(sum(bi*b_0))
        
        :param enc_seed_i : upk_i
        :param num_instances, num_features : label information
        :return enc_mask_A, enc_mask_b: B', c' denoted in the paper
        """
        seed_i = self.msk.decrypt(enc_seed_i)

        mask_X_i = [[PRF(seed_i, h_index + w_index*num_instances, self.mpk.n) for w_index in range(num_features)] for h_index in range(num_instances)]

        # compute B'[i,j]
        enc_mask_A = None
        t_mask_X_i = transpose(mask_X_i)
        if len(self.list_mask_X) != 0:
            for mask_X_j in self.list_mask_X:
                submask = [[self.mpk.encrypt(space_mapping(sum(map(operator.mul, vector1, vector2)), self.mpk.n)) for vector1 in t_mask_X_i] for vector2 in transpose(mask_X_j)]

                if enc_mask_A is None:
                    enc_mask_A = submask
                else: 
                    enc_mask_A += submask

        self.list_mask_X.append(mask_X_i)

        # compute c'[i]
        enc_mask_b = None
        if self.mask_Y is None:
            #self.mask_Y = [PRF(seed_i, index, self.mpk.n) for index in range(num_instances)]
            self.mask_Y = t_mask_X_i[0]
        else:
            enc_mask_b = [self.mpk.encrypt(space_mapping(sum(map(operator.mul, vector1, self.mask_Y)), self.mpk.n)) for vector1 in t_mask_X_i]

        return enc_mask_A, enc_mask_b


    def protocol_ridge_step2(self, enc_C, enc_d):
        """
        Protocol-ridge_version1 Step2(masked model computation)
        - receive Enc(C), Enc(d) from MLE and decrypt them
        - compute w_tilda and determinant of C(=A*R)
        
        :param enc_C: Enc(C) = Enc(A*R) received from MLE
        :param enc_d: Enc(d) = Enc(b + Ar) received from MLE
        :return w_tilda: inverse(C)*d mod N
        """
        # decrypt Enc(C) and Enc(d)
        dec_C = []
        for line in enc_C:
            dec_C.append([self.msk.decrypt(x) for x in line])

        dec_d = [self.msk.decrypt(y) for y in enc_d]

        # compute w_tilda( = inverse(C)*d) using gaussian Elimination algorithm
        temp = [dec_C[index] + [dec_d[index]] for index in range(len(dec_C))]
        w_tilda = gaussElimination(temp, self.mpk.n)

        for index, item in enumerate(w_tilda):
            w_tilda[index] = space_mapping(item, self.mpk.n)

        return w_tilda

    def get_MPK(self):
        return self.mpk
