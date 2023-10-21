from typing import List, Dict, Tuple

import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity as sim_func
import scipy
import networkx as net
import itertools

from vocabulary import Lexicon
from vocabulary import TfidfLexicon, GloveLexicon, TransformersLexicon


def _sentence_sim(sent1: ndarray, sent2: ndarray) -> float:
    """
    Transform the sentences to vector representations and compute the cosine similarity between the sentences
    :param sent1: mean repr
    :param sent2: mean repr
    :return:
    """
    sim_score = sim_func(sent1, sent2)
    return sim_score.item()


def sort_coo(coo_matrix: scipy.sparse.coo.coo_matrix) -> List:
    """
    Sort the tf-idf vectors by descending order of scores.
    coo_matrix is the transformation matrix of the given document
    """
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names: List, sorted_items: List, num=10) -> Dict:
    """
    Get the feature names and tf-idf score of top n items
    """

    # use only top n items from vector
    sorted_items = sorted_items[:num]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results



