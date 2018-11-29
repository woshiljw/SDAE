import numpy as np

class Noise(object):
    def __init__(self,X):
        self.X = X
    def GaussianNoise(self,sd=0.5):
        self.X += np.random.normal(0,sd,self.X.shape)
        return self.X