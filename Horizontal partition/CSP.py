import numpy as np
from paillier import generate_keypair
from util import space_mapping, gaussElimination
from copy import deepcopy


#Crypto Service Provider
class CSProvider:
    def __init__(self):
        self.pk = self.sk = None

    # Protocol-Horizontal Step1(key generation)
    def key_gen(self, n_length):
        self.pk, self.sk = generate_keypair(n_length = n_length)

    def protocol_ridge_step2(self, enc_C, enc_d):
        """
        Protocol-ridge Step2(masked model computation)
        - receive Enc(C), Enc(d) from MLE and decrypt them
        - compute w_tilda and determinant of C(=A*R)
        
        :param enc_C: Enc(C) = Enc(A*R) received from MLE
        :param enc_d: Enc(d) = Enc(b + Ar) received from MLE
        :return w_tilda: inverse(C)*d mod N
        """
        # decrypt Enc(C) and Enc(d)
        dec_C = []
        for line in enc_C:
            dec_C.append([self.sk.decrypt(x) for x in line])

        dec_d = [self.sk.decrypt(y) for y in enc_d]

        # compute w_tilda( = inverse(C)*d) using gaussian Elimination algorithm
        temp = [dec_C[index] + [dec_d[index]] for index in range(len(dec_C))]
        w_tilda = gaussElimination(temp, self.pk.n)

        for index, item in enumerate(w_tilda):
            w_tilda[index] = space_mapping(item, self.pk.n)

        return w_tilda


    # To publish a public key to Data Owners and MLE
    def getPKey(self):
        return self.pk





