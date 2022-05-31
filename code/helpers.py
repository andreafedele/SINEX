import copy
import numpy as np

class Helpers:
    def expand_dimension(self, s):
        ''' Expand the spectrogram dimensions for tensorflow predict_on_batch function use 
            input s: spectrogram '''
        return np.expand_dims(copy.deepcopy(s), axis=0)

    # s: spectrogram
    # p: perturbations
    # segs: segments
    def perturb_input(self, s, p, segs):
        ins = np.where(p == 0)[0]  # finds inactive segments (to be silenced)
        sp = copy.deepcopy(s) # makes a deep copy of the input spect
        for i in ins:
            sp[segs == i] = -80 # silence in dB scale
        
        idxs = np.where(sp != -80) # indexes where the image is not set to 0 by the perturbation (remains active)
        return (sp, idxs)

    def predict_similarity(self, f, x, s):
        return f.predict_on_batch([self.expand_dimension(x), self.expand_dimension(s)])[0][0]