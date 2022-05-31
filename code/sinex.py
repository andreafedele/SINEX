import numpy as np
from helpers import Helpers
from skimage.segmentation import felzenszwalb, slic

class Sinex:
    def __init__(self, algo, params, shape):
        self.algo = felzenszwalb if algo == 'felzenszwalb' else slic # segmentation algorithm
        self.aparams = params # algorithm parameters
        self.shape = shape # input shape
        self.Helpers = Helpers()

    def explain(self, f, x, S):
        E = []

        for sidx in range(len(S)):
            si = S[sidx] # support set si
            
            v = self.Helpers.predict_similarity(f, x, si) # calculate initial similarity
            print("Analyzig support set index:", sidx, "Predicted similarity:", v)

            R = self.algo(si.copy(), **self.aparams) # creating segments on support set si input
            nR = np.unique(R).shape[0] # number of segments

            # Initializes current sample contribution
            hi = np.zeros(self.shape)
        
            for i in range(nR):
                zeros = np.full(nR, 0) # creating array of 0s (inactive segments)
                zeros[i] = 1 # turning on only the current ith segment 
                zi, idxs = self.Helpers.perturb_input(si, zeros, R) # perturb si leaving only current ith segment turned on
                pxl = zi[idxs] # n. of pixels remaining active
                u = self.Helpers.predict_similarity(f, x, zi) # calculate similarity
                d = v - u # calculate delta of similarity scores
                hi[idxs] = d / len(pxl) # weights and updates contribution values using idxs of the only active segment (the ith in exam)

            # Appending current support set attribution's vector
            E.append(hi)
        
        return E