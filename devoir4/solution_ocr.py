# -*- coding: utf-8 -*-

#####
# leba3207
####

from pdb import set_trace as dbg  # Utiliser dbg() pour faire un break dans votre code.

import numpy as np


def logistic(x):
    return 1. / (1. + np.exp(-x))


class ReseauDeNeurones:

    def __init__(self, alpha, T):
        self.alpha = alpha
        self.T = T

    def initialisation(self, W, w):
        self.W = W
        self.w = w

    def parametres(self):
        return self.W, self.w

    def prediction(self, x):
        scores = np.dot(self.W, x)  # TODO: add bias?
        sum_scores = np.sum(scores)
        h_t = logistic(sum_scores)
        prediction = 1 if h_t >= 0.5 else 0
        return prediction

    def mise_a_jour(self, x, y):
        h_t = logistic(x)
        self.W += self.alpha * (y - h_t) * x

    def entrainement(self, X, Y):
        for t in range(self.T):
            for index, (x_t, y_t) in enumerate(zip(X, Y)):
                scores = np.dot(self.W, x_t)  # TODO: add bias?
                sum_scores = np.sum(scores)
                h_t = logistic(sum_scores)
                prediction = 1 if h_t >= 0.5 else 0
                if prediction != y_t:
                    self.mise_a_jour(x_t, y_t)
