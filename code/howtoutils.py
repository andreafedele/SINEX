import os
import glob
import pickle
import random
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class HowToUtils:
    def __init__(self, src, test_classes):
        self.src = src
        self.test_classes = test_classes
        self.shapcmap = LinearSegmentedColormap.from_list('shaplike_gradient', (
            # Edit this gradient at https://eltos.github.io/gradient/#1E88E5-91C5F2-FFFFFF-FF6395-FF0052
            (0.000, (0.118, 0.533, 0.898)),
            (0.250, (0.569, 0.773, 0.949)),
            (0.500, (1.000, 1.000, 1.000)),
            (0.750, (1.000, 0.388, 0.584)),
            (1.000, (1.000, 0.000, 0.322)))
        )

    # --- ONE SHOT BATCH GENERATION RELATED FUNCTIONS --

    def open_pickle_file(self, filepath):    
        file = open(filepath, 'rb')
        return pickle.load(file)

    def get_other_classes(self, val):
        c = self.test_classes.copy()
        c.remove(val)
        c.sort() # sort so to have them always returned in the same order
        return c

    def get_one_shot_batch(self, xclass):
        X, S, labels = [], [], []

        # Selecting a random spectrogram for the current class (query spect x)
        xspects = glob.glob(os.path.join(self.src, xclass, "*pickle"))
        xs_idx = random.randint(0, len(xspects) - 1)
        xpath = xspects[xs_idx]
        x = self.open_pickle_file(xpath)
        xspects.remove(xpath)

        # Selecting another random spectrogram from query x class 
        xx_idx = random.randint(0, len(xspects) - 1)
        xxpath = xspects[xx_idx]
        xx = self.open_pickle_file(xxpath)

        # Appending x and xx from the same class
        X.append([x, xx])
        S.append(xx)
        labels.append([xclass, xclass])

        # Appending different categories's spectrograms, 1 per xclass per each of the xclass of test set
        dcs = self.get_other_classes(xclass)
        
        for dc in dcs:
            dspects = glob.glob(os.path.join(self.src, dc, "*pickle"))
            ds_idx = random.randint(0, len(dspects) - 1)
            ds = self.open_pickle_file(dspects[ds_idx])

            # Appending query spect x and spectrogram from a different class
            X.append([x, ds])
            S.append(ds)
            labels.append([xclass, dc])
        
        return (x, np.array(S), np.array(X), np.array(labels))

    # --- VISUALIZATION RELATED FUNCTIONS --- #

    def get_absolute_max(self, attrs):
        abs_maxs = -float('inf')

        for attr in attrs:
            amax = np.max(np.abs(attr))
            if amax >= abs_maxs:
                abs_maxs = amax
    
        return abs_maxs

    def get_shaplike_cmap(self):
        return self.shapcmap