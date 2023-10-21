from typing import List, Dict, Tuple

import numpy as np
import scipy
from sklearn.metrics.pairwise import cosine_similarity as sim_func
from rouge import Rouge
from vocabulary import Lexicon


def token_overlap(ref_tokens: List, hypo_tokens: List) -> float:
    """
    Calculate the recall of overlap tokens between reference and hypothesis
    :param ref_tokens:
    :param hypo_tokens:
    :return:
    """
    total_num = len(ref_tokens)
    if total_num == 0:
        return 0.
    else:
        overlaps = set(ref_tokens).intersection(set(hypo_tokens))
        return len(overlaps)/total_num


def rouge1(ref_string: str, hypo_string: str) -> Dict:
    """
    Calculate one gram match between reference and hypothesis strings
    :param ref_string:
    :param hypo_string:
    :return:
    """
    func = Rouge()
    result = func.get_scores(hypo_string, ref_string)
    return result[0]['rouge-1']


def facet_accuracy(support_indices: Dict, extracted_indices: List) -> float:
    """

    :param support_indices:
    :param extracted_indices:
    :return:
    """
    found = 0.
    for idx, supported in enumerate(support_indices):
        if set(supported).intersection(set(extracted_indices)):
            found += 1

    return found/len(support_indices)




