import random
import numpy as np
import multiprocessing as multi
from copy import deepcopy 
from util import compute_det, space_mapping, rationalRecon, transpose
from operator import mul
from itertools import chain


#Machine Learning Engine
class MLEngine:
    def __init__(self):
        self.pk = None
        self.merged_enc_A = []
        self.merged_enc_b = []
        self.R = []
        self.r = []
        self.det_R = None
        self.interval = None


    def set_PK(self, pk):
        self.pk = pk


    def add_enc_Ab(self, enc_Ai, enc_bi, lamda, magnitude):
        """
        Protocol-Horizontal Step3(dataset_merge)
        results merge into self.merged_enc_A, self.merged_enc_b

        :para enc_Ai, enc_bi: encrypted data recieved from DOwner_i
        """
        # if a MLE gets enc_A from the first DOwner 
        if len(self.merged_enc_A) == 0:   
            self.merged_enc_A = enc_Ai

            # A = X*X + lamda*I
            lamda = (int)(lamda * (magnitude**2))
            for index in range(len(self.merged_enc_A)):   
                self.merged_enc_A[index][index] += lamda
        else:
            for h_index in range(len(self.merged_enc_A)):
                for w_index in range(h_index, len(self.merged_enc_A[0])):
                    if w_index == h_index:
                        self.merged_enc_A[h_index][w_index] += enc_Ai[h_index][w_index]
                    else:
                        # To improve efficiency, compute a triangular matrix of A
                        # Because A_ij = A_ji
                        temp = self.merged_enc_A[h_index][w_index] + enc_Ai[h_index][w_index]
                        self.merged_enc_A[h_index][w_index] = temp
                        self.merged_enc_A[w_index][h_index] = temp
                        
        # if a MLE gets enc_b from the first DOwner
        if len(self.merged_enc_b) == 0:    
            self.merged_enc_b = enc_bi
        else:
            for w_index in range(len(self.merged_enc_b)):
                self.merged_enc_b[w_index] += enc_bi[w_index]


    def protocol_ridge_step1(self):
        """
        Protocol-ridge Step1(data masking)
        - sample a random matrix(R) and a random vector(r)
        - mask a merged dataset(A, b) with R, r

        :return enc_C: Enc(C) = Enc(A*R)
        :return enc_d: Enc(d) = Enc(b + Ar)
        """
        num_dimension = len(self.merged_enc_A)

        # sample a random matrix(R) and a random vector(r)
        Range = self.pk.n-1
        MaxInt = self.pk.max_int
        R = [[( random.randrange(Range) - MaxInt ) for _ in range(num_dimension)] for _ in range(num_dimension)]
            
        # check that R is invertible. ([det(A)] is non-zero <=> A is invertible)
        # if R is not invertible, random-sample again until R is invertible.
        det_R = compute_det(R, self.pk.n)
        while(det_R == 0.0):
            R = [[( random.randrange(Range) - MaxInt ) for _ in range(num_dimension)] for _ in range(num_dimension)]
            det_R = compute_det(R, self.pk.n)

        r = [(int)( random.randrange(Range) - MaxInt ) for _ in range(num_dimension)]

        self.R = R
        self.r = r
        self.det_R = det_R

        # Matrix multiplication with multi proccessing.
        splitedAR = []
        R_trans = transpose(self.R)
        for i in range(num_dimension):
            for j in range(num_dimension):
                splitedAR.append([self.merged_enc_A[i], R_trans[j]])

        splitedbA = list(zip(self.merged_enc_b, self.merged_enc_A))

        with multi.Pool(processes = multi.cpu_count()) as pool: #multi processing
            # masking C = A*R with multi-processing
            enc_C = pool.map(self.compute_enc_C, splitedAR)

            # masking d = b + A*r
            enc_d = pool.map(self.compute_enc_d, splitedbA)
        pool.close()
        pool.terminate()
        enc_C = list(zip(*[iter(enc_C)]*num_dimension)) # reshape

        '''
        # Matrix multiplication with multi proccessing.
        interval = num_dimension//5
        self.interval = interval
        A_splited = [self.merged_enc_A[i:i + interval] for i in range(0, num_dimension, interval)]
        R_trans = transpose(R)
        R_t_splited = [R_trans[i:i + interval] for i in range(0, num_dimension, interval)]
        splitedAR = []
        for A_i in A_splited:
            for R_t_i in R_t_splited:
                splitedAR.append([A_i, transpose(R_t_i)])

        splitedbA = list(zip(self.merged_enc_b, self.merged_enc_A))

        with multi.Pool(processes = multi.cpu_count()) as pool: #multi processing
            # masking C = A*R with multi-processing
            temp_enc_C = pool.map(self.compute_enc_C, splitedAR)
            enc_C = []
            for i in range(5): #reshape
                enc_C += [list(chain.from_iterable(items)) for items in zip(*temp_enc_C[i:i+5])]

            # masking d = b + A*r
            enc_d = pool.map(self.compute_enc_d, splitedbA)
        pool.close()
        pool.terminate()
        '''

        return enc_C, enc_d


    def compute_enc_C(self, vectors):
        #result = strassenR(vectors[0], vectors[1], len(vectors[0][0]), self.pk.n)
        #return [result[i][:self.interval] for i in range(self.interval)]
        return sum(map(mul, vectors[0], vectors[1]))

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
            w_dash.append(space_mapping((sum(map(mul, self.R[index], w_tilda)) - self.r[index]) , self.pk.n))

        # rational reconstruction
        w_star = rationalRecon(w_dash, self.pk.n)
        w_star = np.asarray(w_star)

        return w_star


