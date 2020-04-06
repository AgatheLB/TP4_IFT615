# -*- coding: utf-8 -*-

#####
# leba3207
####

from pdb import set_trace as dbg  # Utiliser dbg() pour faire un break dans votre code.

from collections import defaultdict, Counter
from tqdm import tqdm

import numpy as np
import re
import math


# Probabilite: Classe permettant de modéliser les distributions P(C) et P(W|C).
#              Pour ce faire, les dictionnaires 'nbMotsParClasse', 'nbDocsParClasse',
#              'freqWC' doivent être remplis lors de la phase d'entraînement. La variable
#              membre vocabulaire sera automatique affectée après l'appel de la fonction
#              creerVocabulaire, vous n'avez donc pas à la modifiée.
#
#              Au final, lors de la prédiction, P, un objet de la classe 'Probabilite',
#              peut être appelé directement de cette façon : P(C=0) ou bien
#              P(W='allo',C=0,delta=1).
#
class Probabilite():

    def __init__(self):
        # Nb. de mots total dans les documents de la catégorie c.
        self.nbMotsParClasse = defaultdict(lambda: 0.)

        # Nb. de documents de la catégorie c.
        self.nbDocsParClasse = defaultdict(lambda: 0.)

        # Nb. de fois que le mot w apparaît dans les documents de la catégorie c.
        self.freqWC = defaultdict(lambda: 0.)

        # Vocabulaire des mots contenus dans tous les documents.
        self.vocabulaire = []

    def probClasse(self, C):
        nb_doc_c = self.nbDocsParClasse.get(C)
        nb_total_doc = 0
        for classe in self.nbDocsParClasse.keys():
            nb_total_doc += self.nbDocsParClasse.get(C)
        return nb_doc_c/nb_total_doc

    def probMotEtantDonneClasse(self, C, W, delta):
        freq_w_doc_c = self.freqWC.get(W, C)
        nb_mots_c = self.nbMotsParClasse.get(C)
        lissage_denom = delta * (len(self.vocabulaire) + 1)
        return (delta + freq_w_doc_c) / (lissage_denom + nb_mots_c)

    def __call__(self, C, W=None, delta=None):
        if W is None:
            return self.probClasse(C)

        return self.probMotEtantDonneClasse(C, W, delta)


# creerVocabulaire: Fonction qui s'occupe de créer une liste (i.e. un vocabulaire)
#                   des mots fréquents dans le corpus. Un mot est fréquent s'il
#                   apparaît au moins 'seuil' fois.
#
# documents: Liste de string représentant chacune le contenu d'un courriel.
#
# seuil: Fréquence minimale d'un mot pour qu'il puisse faire parti du vocabulaire.
#
# retour: Un 'set' contenant l'ensemble des mots ('str') du vocabulaire.
#
def creerVocabulaire(documents, seuil):
    vocabulaire = set()
    for document in tqdm(documents):
        words_doc = document.split()
        for word in words_doc:
            if word not in vocabulaire and words_doc.count(word) >= seuil:
                vocabulaire.add(word)
    return vocabulaire


# pretraiter: Fonction qui remplace les mots qui ne font pas parti du vocabulaire
#             par le token 'OOV'.
#
# doc: Un document représenté sous la forme d'une string.
#
# V: Vocabulaire représenté par un 'set' de mots ('str').
#
# retour: Une 'list' des mots contenu dans le document et présent dans le vocabulaire.
#
def pretraiter(doc, V):
    traited_doc = list()
    words = doc.split()
    for word in words:
        if word in V:
            traited_doc.append(word)
        else:
            traited_doc.append('OOV')
    return doc.split()


# entrainer: Fonction permettant d'entraîner les distributions P(C) et P(W|C)
#            à partir d'un ensemble de courriels.
#
# corpus: Liste de tuples, où chaque tuple est composé d'une liste de
#         mots (i.e. document prétraité) et un entier indiquant la
#         classe (0:SPAM,1:HAM). Par exemple,
#         corpus == [..., (["Mon", "courriel", "..."], 1), ...]
#
# P: Objet de la classe Probabilite qui doit être modifié directement (Référence!)
#
# retour: Rien! L'objet P doit être modifié via ses dictionnaires.
#
def entrainer(corpus, P):
    all_words_0, all_words_1 = list(), list()
    nb_docs_0, nb_docs_1 = 0, 0
    for (x, y) in corpus:
        if y == 0:
            all_words_0.extend(x)
            nb_docs_0 += 1
        else:
            all_words_1.extend(x)
            nb_docs_1 += 1

    counter_0 = Counter(all_words_0)
    counter_1 = Counter(all_words_1)

    P.nbMotsParClasse[0] = len(counter_0)
    P.nbMotsParClasse[1] = len(counter_1)

    P.nbDocsParClasse[0] = nb_docs_0
    P.nbDocsParClasse[1] = nb_docs_1

    for word, count in counter_0.items():
        P.freqWC[(word, 0)] = count
    for word, count in counter_1.items():
        P.freqWC[(word, 1)] = count


# predire: Fonction utilisée pour trouver la classe la plus probable à quelle
#          appartient le document d à partir des distribution P(C) et P(W|C).
#
# doc: Un document représenté sous la forme d'une 'list' de mots ('str').
#
# P: Objet de la classe Probabilite.
#
# C: Liste des classes possibles (0:SPAM,1:HAM)
#
# delta: Paramètre utilisé pour le lissage.
#
# retour: Un tuple (int,float) où l'entier désigne la classe la plus probable
#         du document et le nombre à virgule est la log-probabilité conjointe
#         d'un document D=[w_1,...,w_d] et de catégorie c, i.e. P(C=c,D=[w_1,...,w_d]). *N'oubliez pas vos logarithmes!
#
def predire(doc, P, C, delta):
    prob_jointe = dict()

    for c in C:
        prob_a_priori = P.probClasse(c)
        sum_prob_conjointe = 0
        for word in doc:
            sum_prob_conjointe += P.probMotEtantDonneClasse(c, word, delta)

        prob_jointe[c] = prob_a_priori * sum_prob_conjointe

    best_class = max(prob_jointe, key=prob_jointe.get)
    return best_class, prob_jointe[best_class]
