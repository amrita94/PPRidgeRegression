import numpy as np
import random
from operator import mul, add
import multiprocessing as multi
from copy import deepcopy
from util import compute_det, space_mapping, rationalRecon, transpose #,strassenR
from math import floor, sqrt, ceil
import time 

#Machine Learning Engine
class MLEngine:
    def __init__(self):
        self.mpk = None
        self.merged_enc_A = []
        self.merged_enc_b = []
        self.list_labEnc_Xj = []
        self.labEnc_Y = []
        self.R = []
        self.r = []
        self.det_R = None


    def set_MPK(self, mpk):
        self.mpk = mpk


    def merge_data(self, enc_Ai, enc_bi, labEnc_Xi, labEnc_Y, enc_mask_A, enc_mask_b, lamda, magnitude):
        encoded_lamda = (int)(lamda * (magnitude**2))

        # merged Enc(A)
        if len(self.merged_enc_A) == 0:    # if a MLE gets data from the first DOwner
            assert enc_mask_A is None and enc_mask_b is None
            assert enc_bi is not None
            self.merged_enc_A = self.diagonal_block(enc_Ai, encoded_lamda)
            self.merged_enc_b = enc_bi
        else:

            '''
            # single process
            new_diagonal_block = self.diagonal_block(enc_Ai, encoded_lamda)
            new_nondiagonal_block = self.nondiagonal_block(labEnc_Xi, enc_mask_A)
            '''
            # multi processing
            # parallization with multi-processing to improve the performance
            new_diagonal_block = self.diagonal_block(enc_Ai, encoded_lamda)
            with multi.Pool(processes = multi.cpu_count()) as workers:#multi processing
                new_nondiagonal_block = self.nondiagonal_block_mp(workers, labEnc_Xi, enc_mask_A)
                new_enc_bi = workers.map(self.compute_new_enc_bi, list(zip(enc_mask_b, transpose(labEnc_Xi))))
            workers.close()
            workers.terminate()

            # add blocks to right side
            for index, new_subline in enumerate(new_nondiagonal_block):
                self.merged_enc_A[index] += new_subline

            # add blocks to bottom side
            t_newNonDiagonalBlock = transpose(new_nondiagonal_block)

            for h_index in range(len(enc_Ai)):
                self.merged_enc_A.append(t_newNonDiagonalBlock[h_index] + new_diagonal_block[h_index])

            self.merged_enc_b += new_enc_bi


        # add Enc(Xi), Enc(Yi) on the list
        self.list_labEnc_Xj.append(labEnc_Xi)
        if labEnc_Y is not None:
            self.labEnc_Y = labEnc_Y


    def diagonal_block(self, enc_Ai, encoded_lamda):
        num_dimensions = len(enc_Ai)
        for h_index in range(num_dimensions):
            for w_index in range(h_index, num_dimensions):
                if h_index == w_index:
                    # if (i==j), then add Enc(10^(2*l)*lamda)
                    enc_Ai[h_index][w_index] += encoded_lamda
                else:
                    enc_Ai[w_index][h_index] = enc_Ai[h_index][w_index]

        return enc_Ai


    def nondiagonal_block(self, labEnc_Xi, enc_mask_A):
        new_block = None
        # Matrix multiplication(labEnc_Xj, labEnc_Xi)
        t_labEnc_Xi = transpose(labEnc_Xi)
        for labEnc_Xj in self.list_labEnc_Xj:
            subnew_block = [[sum(map(mul, vector1, vector2)) for vector1 in t_labEnc_Xi] for vector2 in transpose(labEnc_Xj)]
            if new_block is None:
                new_block = subnew_block 
            else: 
                new_block += subnew_block

        # remove mask(sum(Enc(b*b')))
        assert len(new_block) == len(enc_mask_A)
        assert len(new_block[-1]) == len(enc_mask_A[-1])

        new_block = [list(map(add, enc_value, enc_mask)) for enc_value, enc_mask in zip(new_block, enc_mask_A)]

        return new_block

    '''
    # improved version for multi-threading
    def nondiagonal_block_mp(self, workers, labEnc_Xi, enc_mask_A):
        new_block = None
        # Matrix multiplication(labEnc_Xj, labEnc_Xi)
        t_labEnc_Xi = transpose(labEnc_Xi)
        num_dimension = len(t_labEnc_Xi)
        
        for labEnc_Xj in self.list_labEnc_Xj:
            #subnew_block = [[sum(map(mul, vector1, vector2)) for vector1 in t_labEnc_Xi] for vector2 in transpose(labEnc_Xj)]
            subnew_block = workers.map(self.dotproduct, self.splitMatrixs(transpose(labEnc_Xj), t_labEnc_Xi))
            subnew_block = list(zip(*[iter(subnew_block)]*num_dimension)) # reshape
            if new_block is None:
                new_block = subnew_block 
            else: 
                new_block += subnew_block

        # remove mask(sum(Enc(b*b')))
        #assert len(new_block) == len(enc_mask_A) and len(new_block[-1]) == len(enc_mask_A[-1])
        new_block = [list(map(add, enc_value, enc_mask)) for enc_value, enc_mask in zip(new_block, enc_mask_A)]

        return new_block

    #for computing enc_C with multi-processing
    def splitMatrixs(self, mat1, mat2):
        splited = []
        for i in range(len(mat1)):
            for j in range(len(mat2)):
                splited.append([mat1[i], mat2[j]]) 

        return splited
    '''

    # improved version for multi-threading
    def nondiagonal_block_mp(self, workers, labEnc_Xi, enc_mask_A):
        new_block = None
        # Matrix multiplication(labEnc_Xj, labEnc_Xi)
        t_labEnc_Xi = transpose(labEnc_Xi)
        num_dimension = len(t_labEnc_Xi)
        
        for labEnc_Xj in self.list_labEnc_Xj:
            t_labEnc_Xj = transpose(labEnc_Xj)
            splitedmatrice = []
            for vector1 in t_labEnc_Xj:
                for vector2 in t_labEnc_Xi:
                    #splitedmatrice.append([vector1, vector2]) 
                    interval = len(vector1)//10
                    for i in range(0, len(vector1), interval):
                        splitedmatrice.append([vector1[i:i + interval], vector2[i:i + interval]]) 
            temp = workers.map(self.dotproduct, splitedmatrice)
            subnew_block = []
            for i in range(0, len(temp), 10):
                subnew_block.append(sum(temp[i:i + 10]))
            subnew_block = list(zip(*[iter(subnew_block)]*num_dimension)) # reshape
            if new_block is None:
                new_block = subnew_block 
            else: 
                new_block += subnew_block

        # remove mask(sum(Enc(b*b')))
        #assert len(new_block) == len(enc_mask_A) and len(new_block[-1]) == len(enc_mask_A[-1])
        new_block = [list(map(add, enc_value, enc_mask)) for enc_value, enc_mask in zip(new_block, enc_mask_A)]

        return new_block


    #for computing enc_C with multi-processing
    def dotproduct(self, vectors):
        return sum(map(mul, vectors[0], vectors[1]))



    #for computing enc_d with multi-processing
    def compute_new_enc_bi(self, Xm):
        return Xm[0] + sum(map(mul, Xm[1], self.labEnc_Y))


    def protocol_ridge_step1(self):
        """
        Protocol-ridge_version1 Step1(data masking)
        - sample a random matrix(R) and a random vector(r)
        - mask a merged dataset(A, b) with R, r

        :return enc_C: Enc(C) = Enc(A*R)
        :return enc_d: Enc(d) = Enc(b + Ar)
        """
        num_dimension = len(self.merged_enc_A)

        # sample a random matrix(R) and a random vector(r)
        Range = self.mpk.n-1
        MaxInt = self.mpk.max_int
        R = [[( random.randrange(Range) - MaxInt ) for _ in range(num_dimension)] for _ in range(num_dimension)]
            
        # check that R is invertible. ([det(A)] is non-zero <=> A is invertible)
        # if R is not invertible, random-sample again until R is invertible.
        det_R = compute_det(R, self.mpk.n)
        while(det_R == 0.0):
            R = [[( random.randrange(Range) - MaxInt ) for _ in range(num_dimension)] for _ in range(num_dimension)]
            det_R = compute_det(R, self.mpk.n)

        r = [(int)( random.randrange(Range) - MaxInt ) for _ in range(num_dimension)]

        # store R, r in the Object for step3
        self.R = R
        self.r = r
        self.det_R = det_R

        # masking C = A*R with multi-processing
        splitedAR = []
        R_trans = transpose(self.R)
        for i in range(num_dimension):
            for j in range(num_dimension):
                splitedAR.append([self.merged_enc_A[i], R_trans[j]])

        splitedbA = list(zip(self.merged_enc_b, self.merged_enc_A))
        with multi.Pool(processes = multi.cpu_count()) as workers:#multi processing
            # masking C = A*R with multi-processing
            enc_C = workers.map(self.dotproduct, splitedAR)
            # masking d = b + A*r
            enc_d = workers.map(self.compute_enc_d, splitedbA)
        workers.close()
        workers.terminate()

        enc_C = list(zip(*[iter(enc_C)]*num_dimension)) # reshape

        return enc_C, enc_d


    #for computing enc_d with multi-processing
    def compute_enc_d(self, bA):
        return bA[0] + sum(map(mul, bA[1], self.r))


    def protocol_ridge_step3(self, w_tilda):
        """
        Protocol-ridge Step3(model reconstruction)
        - receive w_tilda and det(C)  and compute w*

        :param w_tilda: inverse(C)*d mod N
        :return w_star: Estimated coefficients for the linear regression problem computed by the suggested protocol(version1)
        """

        # compute w_dash = R*w_tilda - r
        w_dash = []
        for index in range(len(w_tilda)):
            w_dash.append(space_mapping((sum(map(mul, self.R[index], w_tilda)) - self.r[index]) , self.mpk.n))

        # rational reconstruction
        w_star = rationalRecon(w_dash, self.mpk.n)
        w_star = np.asarray(w_star)

        return w_star
