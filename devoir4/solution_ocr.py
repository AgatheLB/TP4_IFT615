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
        # TODO: add bias?
        scores_couche_1 = np.dot(self.W, x)
        h_t_couche_1 = logistic(scores_couche_1)

        scores_couche_2 = np.dot(self.w, h_t_couche_1)
        h_t_couche_2 = logistic(scores_couche_2)

        prediction = 1 if h_t_couche_2 >= 0.5 else 0
        return prediction

    def mise_a_jour(self, x, y):
        scores_couche_1 = np.dot(self.W, x)
        h_t_couche_1 = logistic(scores_couche_1)

        scores_couche_2 = np.dot(self.w, h_t_couche_1)
        h_t_couche_2 = logistic(scores_couche_2)

        grad_output = y - h_t_couche_2
        self.w = h_t_couche_1 * (y - h_t_couche_1) * self.w * grad_output

        for i, x_i in enumerate(x):
            self.W[:, i] = x_i * (y - x_i) * self.W[:, i] * self.w



    def entrainement(self, X, Y):
        for t in range(self.T):
            for x_t, y_t in zip(X, Y):
                if self.prediction(x_t) != y_t:
                    self.mise_a_jour(x_t, y_t)
