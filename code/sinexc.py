import numpy as np
from helpers import Helpers
from skimage.segmentation import felzenszwalb, slic

class Sinexc:
    def __init__(self, algo, params, shape, alpha, beta):
        self.algo = felzenszwalb if algo == 'felzenszwalb' else slic # segmentation algorithm
        self.aparams = params # algorithm parameters
        self.shape = shape # input shape
        self.alpha = alpha 
        self.beta = beta
        self.Helpers = Helpers()

    def explain(self, f, x, S):
        E = []

        for sidx in range(len(S)):
            # Gents inputs
            si = S[sidx]
            
            # Stores initial similarity
            v = self.Helpers.predict_similarity(f, x, si) # calculate initial similarity
            print("Analyzig support set index:", sidx, "Predicted similarity:", v)

            # creates segments on support set si input
            R = self.algo(si.copy(), **self.aparams) # creating segments
            nR = np.unique(R).shape[0] # number of segments

            # initializes current sample contribution
            hi = np.zeros(self.shape)
        
            for i in range(nR):
                # fetching current segment pixels to later save contribution values
                zeros = np.full(nR, 0)
                zeros[i] = 1
                _, segidx = self.Helpers.perturb_input(si, zeros, R) 
                
                sims = 0 # initializing similarity scores
                P = np.random.binomial(1, self.beta, size=(self.alpha, nR)) # additional perturbations
                
                for pi in P:
                    pi[i] = 1 # forcing current ith segment as active in current pi_th perturbation
                    zi, idxs = self.Helpers.perturb_input(si, pi, R) # perturb si using pi leaving only current ith segment turned on
                    pixels = zi[idxs] # actual pixels remaining active
                    u = self.Helpers.predict_similarity(f, x, zi) # calculate similarity
                    sims += u # storing u similarity
                
                delta = v - (sims / len(P)) # calculating delta of similarity scores
                hi[segidx] = delta / len(pixels) # weights and updates contribution values using idxs of the ith segment in exam
                
            # Appending current support set attribution's vector
            E.append(hi)
        
        return E