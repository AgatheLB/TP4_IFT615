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
        scores_1 = np.dot(self.W, x)
        h_1 = logistic(scores_1)

        scores_2 = np.dot(self.w, h_1)
        h_2 = logistic(scores_2)

        prediction = 1 if h_2 >= 0.5 else 0
        return prediction

    def mise_a_jour(self, x, y):
        scores_1 = np.dot(self.W, x)
        h_1 = logistic(scores_1)

        scores_2 = np.dot(self.w, h_1)
        h_2 = logistic(scores_2)

        di_2 = y - h_2

        for i in range(self.W.shape[0]):
            di_1 = h_1[i] * (1 - h_1[i]) * self.w[i] * di_2
            for n, x_n in enumerate(x):
                self.W[i, n] += self.alpha * x_n * di_1

        for i, h_i_1 in enumerate(h_1):
            self.w[i] += self.alpha * h_i_1 * di_2




    def entrainement(self, X, Y):
        for t in range(self.T):
            for x_t, y_t in zip(X, Y):
                # if self.prediction(x_t) != y_t: # TODO: to removeha m
                self.mise_a_jour(x_t, y_t)
