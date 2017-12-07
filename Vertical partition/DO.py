#for MLE
import numpy as np
from util import space_mapping
from operator import mul
from labScheme import localGen


#Data Owner
class DOwner:
    def __init__(self, Xi, Yi, magnitude):
        self.num_instances = Xi.shape[0]
        self.num_features = Xi.shape[1]
        self.Xi = Xi
        self.Yi = Yi
        self.mpk = None
        self.magnitude = magnitude
        self.seed = self.enc_seed = None
        self.enc_Ai = None
        self.enc_bi = None
        self.labEnc_Xi = None
        self.labEnc_Yi = None


    def set_MPK(self, mpk):
        self.mpk = mpk
        self.seed, self.enc_seed = localGen(mpk)        # Protocol-Vertical Step1(key setup)


    def Enc_localData(self):
        """
        Protocol-Vertical Step2(local computation)

        :param self.X, self.Y: training dataset. X = (n,d) matrix, Y = n vector
        :param magnitude: parameter for data representation(if we use at most "l" fractional digits, then magnitude = 10**l)
        :return self.enc_A, self.enc_b: Enc(A), Enc(b) 
        :return self.labEnc_Xi, self.labEnc_Yi : labEnc(X), labEnc(Y)
        """

        # convert a data representation domain(real -> Integer)
        # R => Z<l+r> => Z<N>
        Xi = (self.Xi * self.magnitude).astype(int)

        # compute Enc(Ai)
        Xi_trans = np.transpose(Xi)
        Ai = np.matmul(Xi_trans, Xi).astype(int).tolist()
        enc_Ai = []
        for index, line in enumerate(Ai):
            enc_Ai.append([0]*index + [self.mpk.encrypt(space_mapping(item, self.mpk.n)) for item in line[index:]])

        # compute labEnc(Xi)
        Xi = Xi.tolist()
        labEnc_Xi = [[self.mpk.labEncrypt(self.seed, (h_index + w_index*self.num_instances), Xi_item) for Xi_item, w_index in zip(Xi_line, range(self.num_features))] for Xi_line, h_index in zip(Xi, range(self.num_instances))]

        # Enc()
        if self.Yi is None:
            enc_bi = None
            labEnc_Yi = None
        else:
            #Enc(bi) = Enc(Xi_trans, Yi)
            Yi = (self.Yi * self.magnitude).astype(int)
            bi = np.matmul(Xi_trans, Yi).astype(int).tolist()
            enc_bi = [self.mpk.encrypt(space_mapping(item, self.mpk.n)) for item in bi]

            Yi =  Yi.tolist()
            labEnc_Yi = [self.mpk.labEncrypt(self.seed, index, Yi_item) for Yi_item, index in zip(Yi, range(self.num_instances))]

        # return 
        self.enc_Ai = enc_Ai
        self.enc_bi = enc_bi
        self.labEnc_Xi = labEnc_Xi
        self.labEnc_Yi = labEnc_Yi


    # To send enc_A and enc_b to MLE for merging datasets
    def getEnc_AbXY(self):
        return self.enc_Ai, self.enc_bi, self.labEnc_Xi, self.labEnc_Yi

    def get_masking_info(self):
        return self.enc_seed, self.num_instances, self.num_features
