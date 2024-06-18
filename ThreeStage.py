import numpy
from typing import List, Tuple, Dict, Any
import math


class ThreeStage:
    def __init__(self, niter: int, proportions: List[float], final_decay_factor=20):
        proportions = numpy.array(proportions, dtype=float)
        proportions /= proportions.sum() #notrmalize 
        self.bounds = [round(niter * cs) for cs in proportions.cumsum()]
        self.final_decay_factor = final_decay_factor

    def lr(self, iter):
        if iter < self.bounds[0]:
            return iter / self.bounds[0]
        elif iter < self.bounds[1]:
            return 1
        else:
            iter -= self.bounds[1]
            iter /= self.bounds[2] - self.bounds[1] ## relative proportion of final part
            return math.exp(-iter * math.log(self.final_decay_factor))

