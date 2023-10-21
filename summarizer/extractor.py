from typing import List, Dict, Tuple

import numpy as np
import torch
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity as sim_func
import scipy
import networkx as net
import itertools

from torch import Tensor, nn
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from constants import STOPWORDS
from vocabulary import Lexicon
from vocabulary import GloveLexicon, TransformersLexicon
import networkx as nx

import pytorch_lightning as pl
from transformers import BertForSequenceClassification as BertSeqModel
from transformers import (AdamW, WEIGHTS_NAME, get_linear_schedule_with_warmup)

class Extractor(object):
    """
    Basic extractor class
    """


class KeywordsExtractor(Extractor):
    """
    Extractor selecting keywords based on tfidf scores. 
    """

    def __init__(self, corpus: List = None, lowercase: bool = True):
        super().__init__()
        self.lowercase = lowercase
        self._init_vectorizer()
        if corpus is not None:
            self._fit_corpus(corpus)

    def _init_vectorizer(self, stop_words=STOPWORDS, lowercase=None):
        """
        Initialize the tfdif vectorizer from sklearn
        """
        if lowercase:
            self.lowercase = lowercase
        self.vectorizer = TfidfVectorizer(lowercase=self.lowercase, stop_words=stop_words)

    def _fit_corpus(self, corpus: List):
        """
        Build the Lexicon using the given corpus
        """
        self.vectorizer.fit(corpus)
        self.vocabulary = self.vectorizer.get_feature_names()
        self.word_to_idx = {w: i for i, w in enumerate(self.vocabulary)}

    def sort_coo(self, document: str) -> List:
        """
        Sort the tf-idf vectors by descending order of scores.
        coo_matrix is the transformation matrix of the given document
        """
        features = self.vectorizer.transform([document])
        coo_matrix = features.tocoo()
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def _get_keywords_scores(self, document: str, ratio = 0.) -> Dict:
        """
        Extract the top k keywords from one document
        """
        sorted_items = self.sort_coo(document)
        if ratio == 0.:
            #ratio = 0.5
            ratio = 0.08
        #num = int(len(document.split()) * ratio)
        #results = {self.vocabulary[idx]:round(score,3) for idx, score in sorted_items[:num]}
        results = {self.vocabulary[idx]:round(score,3) for idx, score in sorted_items if score > ratio}
        return results

    def _get_sentence_importance(self, sentences: List[str],  key_word_ratio: float = 1.):
        """
        Return the importance of each sentence based on the tfidf value of each token in the sentence
        :param sentences:
        :param key_word_ratio:
        :param threshold:
        :return:
        """
        document = ' '.join(sentences)
        keywords_in_doc = self._get_keywords_scores(document, key_word_ratio)
        if self.lowercase:
            sentences = [sent.lower().split() for sent in sentences]
        else:
            sentences = [sent.split() for sent in sentences]
        sentences_importance = {}

        for i, sent in enumerate(sentences):
            score = 0.
            overlap = set(sent).intersection(set(keywords_in_doc.keys()))
            if len(overlap) > 0:
                # each key word has a tf-idf importance value
                # normalize the key word values along with the amount of key words
                #score = sum([keywords_in_doc[w] for w in overlap]) / len(overlap)
                score = sum([keywords_in_doc[w] for w in overlap])
            sentences_importance[i] = (score, ' '.join(sent))
        return sentences_importance

    def _get_key_sentences(self, sentences: List[str], key_word_ratio: float = 0.8, k:int = None, importance_t: float = None) -> List:
        """
         Extract the key sentences from the document based on keywords.
        :param sentences:
        :param key_word_ratio:
        :return:
        """
        sentences_importance = self._get_sentence_importance(sentences, key_word_ratio)
        if importance_t is not None:
            sentences_importance = {i: v for i, v in sentences_importance.items() if v[0] > importance_t}

        sorted_sentences = sorted(sentences_importance.items(), key=lambda item: item[1][0], reverse=True)
        
        if k is not None:
            resorted = [pair[1][1] for pair in sorted(sorted_sentences[:k])]
        else:
            resorted = [pair[1][1] for pair in sorted(sorted_sentences)]
        return resorted


def _find_closest_args(centroids: ndarray, features: ndarray, threshold=0.1) -> Dict:
    """
    Find the closest arguments to centroid.
    :param centroids: Centroids to find closest.
    :return: Closest arguments.
    """
    args = {i: [] for i in range(len(centroids))}
    for j, centroid in enumerate(centroids):
        # calculate the distance
        values = [np.linalg.norm(feature - centroid) for feature in features]
        args[j] = [idx for idx, dist in enumerate(values) if dist < threshold]
        if len(args[j]) == 0:
            args[j].append(np.argmin(values))
    return args


class CentroidExtractor(Extractor):
    """
    Extractor based on clustering algorithm using bert word embeddings
    """

    def __init__(self, lexicon: Lexicon, ratio: float = None, elbow=True):
        """
        Elbow method for optimal value of k in KMeans based on  inertia.
        Inertia: It is the sum of squared distances of samples to their closest cluster center.
         :param ratio: Ratio to use for clustering.
        """
        super().__init__()
        self.lexicon = lexicon
        self.ratio = 0.6 if ratio is None else ratio
        self.elbow = elbow

    def _get_features(self, sentences: List, with_context=True) -> ndarray:
        """
        Transform sentences strings to features
        :param sentences:
        :return:
        """
        features = self.lexicon.sentences_mean_repr(sentences, with_context)
        features = np.concatenate(features, axis=0)
        return features

    def _calculate_elbow(self, k_max: int, sentences: List) -> List[float]:
        """
        Calculates elbow up to the provided k_max.
        :param k_max: K_max to calculate elbow for.
        :return: The inertia up to k_max.
        """
        inertia = []
        features = self._get_features(sentences)
        for k in range(1, min(k_max, len(features))):
            model = KMeans(k, random_state=42).fit(features)
            inertia.append(model.inertia_)

        return inertia

    def _calculate_optimal_cluster(self, k_max: int, sentences: List):
        """
        Calculates the optimal cluster based on Elbow.
        :param k_max: The max k to search elbow for.
        :return: The optimal cluster size.
        """
        delta_1 = []
        delta_2 = []

        max_strength = 0
        k = 1

        inertia = self._calculate_elbow(k_max, sentences)

        for i in range(len(inertia)):
            delta_1.append(inertia[i] - inertia[i - 1] if i > 0 else 0.0)
            delta_2.append(delta_1[i] - delta_1[i - 1] if i > 1 else 0.0)

        for j in range(len(inertia)):
            strength = 0 if j <= 1 or j == len(inertia) - 1 else delta_2[j + 1] - delta_1[j + 1]

            if strength > max_strength:
                max_strength = strength
                k = j + 1
        return k

    def _top_k_sentences(self, sentences: List, max_k: int = None, threshold=0.07) -> Dict:
        """
        Clusters sentences based on the ratio.
        :param elbow: search best number of cluster
        :param max_k: maximum clusters
        :param sentences: list of sentences. num of sentences overrides ratio.
        :return: Sentences index that qualify for summary.
        """

        num_sentences = len(sentences)
        if num_sentences == 0:
            return {}
        else:
            if max_k is not None:
                k = min(int(num_sentences * self.ratio), max_k)
            else:
                k = max(int(num_sentences * self.ratio), 1)

            features = self._get_features(sentences)

            if self.elbow:
                k = self._calculate_optimal_cluster(k, sentences)

            model = KMeans(k, random_state=42).fit(features)

            centroids = model.cluster_centers_
            # cluster_args = self._find_closest_args(centroids, features)
            cluster_args = _find_closest_args(centroids, features, threshold=threshold)

            # sorted_values = sorted(cluster_args.values())
            # return sorted_values
            top_k = {}
            for i in range(len(cluster_args)):
                top_k[i] = {}
                for s in cluster_args[i]:
                    top_k[i][s] = sentences[s]
            return top_k


def normalize_matrix(S:ndarray):
    """
    :param S:
    :return:
    """
    for i in range(len(S)):
        if S[i].sum() == 0:
            S[i] = np.ones(len(S))

        S[i] /= S[i].sum()
    return S


class TextRankExtractor(Extractor):
    """
    Extractor based on pagerank algorithm and cosine similarity between sentences
    """

    def __init__(self, lexicon: Lexicon, ratio: float = None, token_level: bool = False):
        """
        """
        super().__init__()
        self.lexicon = lexicon
        self.ratio = 1. if ratio is None else ratio
        self.token_level = token_level

    def _get_features(self, sentences: List) -> ndarray:
            """
            Transform sentences strings to features
            :param sentences:
            :return:
            """
            if not self.token_level:
                features = self.lexicon.sentences_mean_repr(sentences)
            else:
                features = self.lexicon.featurize(sentences)
            return features

    def _sim_matrix(self, sentences: List, sentences_importance: Dict = None) -> ndarray:
        """
        Build cross similarity matrix between all sentences
        :param sentences: list of tokens
        :return:
        """
        num_sents = len(sentences)
        features = self._get_features(sentences)
        sim_mat = np.zeros((num_sents, num_sents))
        for i in range(num_sents):
            for j in range(num_sents):
                if i != j:
                    if not self.token_level:
                        sim_mat[i][j] = 1.0 - sim_func(features[i], features[j]).item()
                    else:
                        sim_mat[i][j] = np.sum(sim_func(features[i], features[j])).item()
                    if sentences_importance is not None:
                        sim_mat[i][j] += (sentences_importance[i][0] + sentences_importance[j][0])/2

        return normalize_matrix(sim_mat)

    def _rank_sentences(self, sentences: List, sentences_importance: Dict = None):
        """
        :param sentences:
        :return:
        """
        sim_mat = self._sim_matrix(sentences, sentences_importance)
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i],i, s) for i,s in enumerate(sentences)), reverse=True)
        return ranked_sentences

    def _top_k_sentences(self, sentences: List, k: int, sentences_importance: Dict = None) -> Dict:
        """
        Construct the graph using similarity matrix and find the top k sentences
        :param sentences:
        :return:
        """

        #if max_k is not None:
        #     k = min(int(len(sentences) * self.ratio), max_k)
        #else:
        #    k = max(int(len(sentences) * self.ratio), 1)
        ranked_sentences = self._rank_sentences(sentences, sentences_importance)[:k]
        top_k = {ranked_sentences[i][1]: (ranked_sentences[i][0], ranked_sentences[i][2]) for i in range(len(ranked_sentences))}
        return top_k


class ClassifierExtractor(pl.LightningModule):
    
    def __init__(self, train_config):
        """
        Elbow method for optimal value of k in KMeans based on  inertia.
        Inertia: It is the sum of squared distances of samples to their closest cluster center.
         :param ratio: Ratio to use for clustering.
        """
        super().__init__()
        self.train_config = train_config
        bert_model = train_config['bert_model']
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        if 'fine_tuned' in self.train_config:
            self.model = BertSeqModel.from_pretrained(train_config['fine_tuned'])
        else:
            self.model = BertSeqModel.from_pretrained(bert_model)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor = None):
        """
        Classifier takes sequence inputs and makes binary prediction
        :param input_ids: tokenized source ids
        :param attention_mask:
        :param labels: 1 or 0
        :return:
        """
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        if labels is not None:
            labels = labels.to(self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
        return outputs
    
    def training_step(self, batch, batch_idx):

        """
        :param: batch (sentences, labels)
        """
        #print('model device ', self.model.device)
        
        xs = batch[0]
        ys = batch[1]
        #print('labels type ', type(ys), ys)
        if type(xs) != list:
            xs = list(xs)
        inputs = self.tokenizer(xs, return_tensors='pt', padding=True)
        inputs_to_model_device = {k:inputs[k].to(self.model.device) for k in inputs}
        outputs = self(input_ids = inputs_to_model_device['input_ids'][:,:self.tokenizer.model_max_length], attention_mask = inputs_to_model_device['attention_mask'][:, :self.tokenizer.model_max_length], labels = ys)
        return outputs.loss

    def validation_step(self, batch, batch_idx):

        """
        :param: batch (sentences, labels)
        """
        #print('batch ', batch)
        xs = batch[0]
        ys = batch[1]
        if type(xs) != list:
            xs = list(xs)
        inputs = self.tokenizer(xs, return_tensors='pt', padding=True)
        inputs_to_model_device = {k:inputs[k].to(self.model.device) for k in inputs}
        outputs = self(input_ids = inputs_to_model_device['input_ids'][:,:self.tokenizer.model_max_length], attention_mask = inputs_to_model_device['attention_mask'][:,:self.tokenizer.model_max_length], labels = ys)
        #self.log['valid_loss']
        return outputs.loss

    def configure_optimizers(self):
        """
        :param config: training configs
        :return:
        """
        weight_decay = self.train_config.get("weight_decay", 0.3)
        no_decay = ["bias", "LayerNorm.weight"]
        lr = self.train_config.get('learning_rate', 0.000001)
        betas = self.train_config.get('betas', (0.9, 0.999))

        optimizer_grouped_parameters = [
                {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,},
                
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,},]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas = betas)
        return optimizer

if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader as data_gen
    train_config = {'data_dir': "/gpu/data/OE0441/s460g/",'bert_model': "dbmdz/bert-base-german-cased", 
                    'learning_rate': 0.0001, 'betas': (0.9, 0.999), 'model_dir': '/gpu/checkpoints/OE0441/s460g/extractors1'}
    # train_config = {'data_dir': "",'bert_model': "dbmdz/bert-base-german-cased", 
    #                'learning_rate': 0.000001, 'betas': (0.9, 0.999), 'model_dir': 'extractors'}
    # construct train and valid data
    src_path = os.path.join(train_config['data_dir'], 'extractive_data/extractive_train.txt')
    label_path = os.path.join(train_config['data_dir'], 'extractive_data/extractive_train_label.txt')
    train_dataset = [(s.strip(), int(l.strip())) for s, l in zip(open(src_path).readlines(), open(label_path).readlines())]
    src_path = os.path.join(train_config['data_dir'], 'extractive_data/extractive_valid.txt')
    label_path = os.path.join(train_config['data_dir'], 'extractive_data/extractive_valid_label.txt')
    valid_dataset = [(s.strip(), int(l.strip())) for s, l in zip(open(src_path).readlines(), open(label_path).readlines())]
    
    train_loader = data_gen(train_dataset, 20, True)
    valid_loader = data_gen(valid_dataset, 20, True)

    # construct the classifier
    #train_config = {'data_dir': "/gpu/data/OE0441/s460g/",'bert_model': "dbmdz/bert-base-german-cased", 'learning_rate': 0.000001, 'betas': (0.9, 0.999), 'model_dir': 'extractors'}
    classifier = ClassifierExtractor(train_config)

    # construct the trainer
    trainer = pl.Trainer(gpus=1, gradient_clip_val = 1.0,stochastic_weight_avg=True, max_epochs=10)
    trainer.fit(classifier, train_loader, valid_loader)

    classifier.model.save_pretrained(train_config['model_dir'])

    trainer.save_checkpoint(train_config['model_dir']+ '/best.ckpt')

    del classifier, trainer
    torch.cuda.empty_cache()
















