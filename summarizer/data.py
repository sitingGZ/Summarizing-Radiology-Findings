from typing import List, Dict, Tuple

import os
import pandas as pd
import numpy as np
from numpy import ndarray
import re
import collections
from sklearn.metrics.pairwise import cosine_similarity as sim_func

from constants import HIGH_OVERLAP_RATIO, LOW_OVERLAP_RATIO, MIDDLE_RANGE_RATIO
from metric import token_overlap


def sentence_separators(text, p=r'\s[.*]+\.\s[A-Z]+[a-zäöüß]+') -> List:
    pattern = re.compile(p)
    separators = pattern.findall(text)
    return separators


def get_sentences(text) -> List:
    """
    Split text into sentences based on the regular expression matched in each text.
    # exclusions = ['n. ', 'bzw. ', 'Z. n' ... ]
    """
    text = text.replace('"', '')
    text = text.replace('Z. n.', 'Z.n.')
    sentences = re.split(r'(?<=.[.?]) +(?=[A-Z])|(?<=.[.?])+(?=[A-Z])', text)
    rejoin_sentences = []
    cur_s = sentences[0]
    for i in range(1, len(sentences)):
        if re.match(r'[^A-Za-z0-9][a-zA-Z]|[^A-Za-z0-9][0-9]', cur_s[-3:-1]):
            cur_s += ' '+sentences[i]
            i += 1

        else:
            rejoin_sentences.append(cur_s)
            cur_s = sentences[i]

    if cur_s not in rejoin_sentences:
        rejoin_sentences.append(cur_s)

    sentences = [s[:-1] +' .' if s[-1] == '.' else s + ' .' for s in rejoin_sentences ]

    return sentences


def sentences_to_tokens(sentences: List) -> List:
    return ' '.join(sentences).split()


def make_train_corpus(train_sets) -> List:
    """
    Use the training data excluding the targets to construct a training corpus
    """
    corpus = [' '.join(sents[0] + sents[1]) for sents in train_sets]
    return corpus


def sentences_should_in_summary(dataset):
    general = []
    findings = []
    for i, sample in enumerate(dataset):
        gen_tokens = [l.split() for l in sample[0]]
        find_tokens = [l.split() for l in sample[1]]
        ref_tokens = ' '.join(sample[2]).split()
        labels_g = np.array([ s  for s, hypo_tokens in enumerate(gen_tokens) if len(set(ref_tokens).intersection(set(hypo_tokens))) > 2])
        labels_f = np.array([s  for s, hypo_tokens in enumerate(find_tokens) if len(set(ref_tokens).intersection(set(hypo_tokens))) > 2])
        general.append(labels_g)
        findings.append(labels_f)
    return general, findings


class ReportData(object):
    """
    Read source files and target files from local directory
    """
    def __init__(self, tsv_path:str):
        """
        Loader for constructing train, valid and test data.
        The sources files come from the general sources and finding sources.
        """
        data_df = pd.read_csv(tsv_path, sep='\t', header=0)
        duplicates = list(data_df.duplicated(subset='beurteilung', keep=False))
        self.duplicates = [i for i, b in enumerate(duplicates) if b is True ]

        self.general_documents = [line for line in data_df['allgemein']]
        self.finding_documents = [line for line in data_df['befund']]
        self.target_documents = [line for line in data_df['beurteilung']]

    def split_train_test(self, split) -> Tuple:
        """
        Split the data into train, validate and test sets according to the split ratio.
        split default ratio =  [0.8, 0.1, 0.1]
        """
        
        total_num = len(self.general_documents)
        train_num = int(total_num * split[0])
        valid_num = int(total_num * split[1])

        train_indices = self.duplicates
        remain = list(set(range(total_num)).difference(set(self.duplicates)))
        train_indices += [remain[i] for i in range(train_num - len(self.duplicates))]
        remain = list(set(range(total_num)).difference(set(train_indices)))
        valid_indices = [remain[i] for i in range(valid_num)]
        test_indices = [i for i in remain if i not in valid_indices]

        train_sets = [(get_sentences(self.general_documents[i]), get_sentences(self.finding_documents[i]),
                       get_sentences(self.target_documents[i])) for i in train_indices]
        valid_sets = [(get_sentences(self.general_documents[i]), get_sentences(self.finding_documents[i]),
                       get_sentences(self.target_documents[i])) for i in valid_indices]
        test_sets = [(get_sentences(self.general_documents[i]), get_sentences(self.finding_documents[i]),
                      get_sentences(self.target_documents[i])) for i in test_indices]

        return train_sets, valid_sets, test_sets

    def documents_to_sentences(self) -> Tuple:
        """
        Convert all three documents to sentences
        :return:
        """
        general_sentences = [get_sentences(doc) for doc in self.general_documents]
        finding_sentences = [get_sentences(doc) for doc in self.finding_documents]
        target_sentences = [get_sentences(doc) for doc in self.target_documents]
        return general_sentences, finding_sentences, target_sentences

    def compute_token_overlap(self) -> Tuple:
        """
        Compute the token overlap ratio between general, finding and target documents.
        Documents have to be converted to sentences, rejoined and split into tokens
        :return:
        """
        general_tokens = [sentences_to_tokens(get_sentences(doc)[:2]) for doc in self.general_documents]
        finding_tokens = [sentences_to_tokens(get_sentences(doc)) for doc in self.finding_documents]
        target_tokens = [sentences_to_tokens(get_sentences(doc)) for doc in self.target_documents]

        token_overlap_ratio_general = [token_overlap(target_tokens[i], general_tokens[i]) for i in range(len(target_tokens))]
        token_overlap_ratio_finding = [ token_overlap(target_tokens[i], finding_tokens[i]) for i in range(len(target_tokens))]
        return token_overlap_ratio_general, token_overlap_ratio_finding

    def get_high_low_overlap_test_data(self) -> Tuple:
        """
        High ratio and low ratio are calculated based on the overlaps inspect,
        Refer to vocabulary.ipynb for more details.
        :return:
        """
        train_sets = []
        valid_sets = []
        test_sets_high = {r: [] for r in HIGH_OVERLAP_RATIO}
        test_sets_low = {r: [] for r in LOW_OVERLAP_RATIO}
        test_sets = []
        token_overlap_ratio_general, token_overlap_ratio_finding = self.compute_token_overlap()
        token_overlap_ratio = [round(r1+r2,2) for r1, r2 in zip(token_overlap_ratio_general, token_overlap_ratio_finding)]
        general_sentences, finding_sentences, target_sentences = self.documents_to_sentences()
        for i,r in enumerate(token_overlap_ratio):
            if r in test_sets_high:
                if len(test_sets_high[r]) < HIGH_OVERLAP_RATIO[r] and i not in self.duplicates:
                    test_sets_high[r].append((general_sentences[i], finding_sentences[i], target_sentences[i]))
                    test_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))
                else:
                    if len(valid_sets) <= int(len(token_overlap_ratio) * 0.1) and i not in self.duplicates:
                        valid_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))
                    else:
                        train_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))
            elif r in test_sets_low:
                if len(test_sets_low[r]) < LOW_OVERLAP_RATIO[r] and i not in self.duplicates:
                    test_sets_low[r].append((general_sentences[i], finding_sentences[i], target_sentences[i]))
                    test_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))
                else:
                    if len(valid_sets) <= int(len(token_overlap_ratio) * 0.1) and i not in self.duplicates:
                        valid_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))
                    else:
                        train_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))
            else:
                if len(valid_sets) <= int(len(token_overlap_ratio) * 0.1):
                    valid_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))
                else:
                    train_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))

        return train_sets, valid_sets, test_sets, test_sets_high, test_sets_low

    def get_middle_range_test_data(self) -> Tuple:
        """
        :return:
        """
        train_sets = []
        valid_sets = []
        test_sets = []

        token_overlap_ratio_general, token_overlap_ratio_finding = self.compute_token_overlap()
        token_overlap_ratio = [round(r1+r2, 2) for r1, r2 in zip(token_overlap_ratio_general, token_overlap_ratio_finding)]
        general_sentences, finding_sentences, target_sentences = self.documents_to_sentences()

        for i, r in enumerate(token_overlap_ratio):
            if MIDDLE_RANGE_RATIO[0] <= r <= MIDDLE_RANGE_RATIO[1] and len(test_sets) <= int(len(token_overlap_ratio) * 0.1):
                if i not in self.duplicates:
                    test_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))
                else:
                    train_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))

            else:
                if len(valid_sets) <= int(len(token_overlap_ratio) * 0.1):
                    if i not in self.duplicates:
                        valid_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))
                    else:
                        train_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))

                else:
                    train_sets.append((general_sentences[i], finding_sentences[i], target_sentences[i]))

        return train_sets, valid_sets, test_sets



        






