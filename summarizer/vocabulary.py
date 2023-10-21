from typing import List, Dict, Tuple

import numpy as np
from numpy import ndarray
import scipy
from torch import Tensor
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from constants import STOPWORDS
from transformers import AutoConfig, AutoTokenizer, AutoModel


class Lexicon:
    """
    Base Lexicon calss
    """
    
    @property
    def vocab_size(self) -> int:
        return 0

    @property
    def dim_size(self)-> int:
        return 0

    def featurize(self, *args):
        raise NotImplementedError


class TfidfLexicon(Lexicon):
    """
    Create the tf-idf Lexicon and vectors
    """
    def __init__(self,corpus, stop_words=STOPWORDS, lowercase=True):
        super().__init__()
        """
        Vocaublary built by tfidf vectorizer.
        """
        self._init_vectorizer(stop_words,lowercase)
        self._fit_corpus(corpus)
        self.doc_size = len(corpus)

    def _init_vectorizer(self, stop_words, lowercase=True):
        """
        Initialize the tfdif vectorizer from sklearn
        """
        self.vectorizer = TfidfVectorizer(lowercase = lowercase, stop_words= stop_words)

    def _fit_corpus(self, corpus: List):
        """
        Build the Lexicon using the given corpus
        """
        self.vectorizer.fit(corpus)
        self.vocabulary = self.vectorizer.get_feature_names()
        self.word_to_idx = {w:i for i,w in enumerate(self.vocabulary)}

    def featurize(self, sentences: List) -> scipy.sparse.csr.csr_matrix:
        """
        Transform the document into vectors with the dim of vocabulary length
        """
        document = ' '.join(sentences)
        features = self.vectorizer.transform([document])
        return features

    def dim_size(self):
        """
        Return the features dimension
        """
        return self.doc_size

    def vocab_size(self):
        """
        Return the size of the vocabulary
        """
        return len(self.vocabulary)


class GloveLexicon(Lexicon):
    """Lexicon built with saved glove vector and vocab text files."""

    def __init__(self, vectors_path):
        super().__init__()

        self.vector_lines = [line.strip().split() for line in open(vectors_path)]
        self.word_to_idx = {line[0]: i for i, line in enumerate(self.vector_lines)}
        
    def featurize(self, sentences : List) -> List:
        """
        Transform the words in the list into vectors with the dim of Lexicon length
        """
        if type(sentences[0]) == str:
            sentences = [sent.split() for sent in sentences]

        features = []
        for i, bag_of_words in enumerate(sentences):
            features.append([self._look_up(w) for w in bag_of_words if w.lower() in self.word_to_idx])
        #features[i] = np.zeros(self.dim_size(), dtype='float32')
        return features

    def sentences_mean_repr(self, sentences: List) -> List:
        """
        Return the mean features of sentences based on word vectors.
        :param words_features:
        :return:
        """
        sent_features = []
        #if type(sentences[0]) == str:
        #    sentences = [s.split() for s in sentences]

        features = self.featurize(sentences)
        for feature in features:
            if len(feature) > 0:
                #feature = np.concatenate(feature, axis=0)
                sent_features.append(np.expand_dims(np.mean(feature, axis=0),axis=0))
            else:
                sent_features.append(np.zeros((1, self.dim_size())))
        return sent_features

    def _get_vocabulary(self):
        self.vocabulary = [line[0] for line in self.vector_lines]

    def _look_up(self,w:str) -> ndarray:
        """
        Return the w vector from the vocabulary
        """
        idx = self.word_to_idx[w.lower()]
        return np.asarray(self.vector_lines[idx][1:], dtype='float32')

    def dim_size(self):
        """
        Return the features dimension
        """
        return len(self.vector_lines[0][1:])

    def vocab_size(self):
        """
        Return the size of the vocabulary
        """
        return len(self.vector_lines)

    def __repr__(self):
        return "GloveLexicon"


class TransformersLexicon(Lexicon):
    """Language models are BERT-based."""

    def __init__(self, model_path, max_position_length: int = 512):
        super().__init__()
        """
        Default model path : "german bert"
        """
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

        #self.model = self._set_model_weight()
        #self._set_model_weight()
        self.vectors = self.model.get_input_embeddings().weight
        if not self.tokenizer.eos_token:
            self.tokenizer.eos_token = self.tokenizer.pad_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not self.tokenizer.cls_token:
            self.tokenizer.cls_token = self.tokenizer.eos_token
        if not self.tokenizer.sep_token:
            self.tokenizer.sep_token = self.tokenizer.eos_token

    def model_max_length(self, new_length = 1000):
        """
        :param new_length:
        :return:
        """
        self.config.max_position_embeddings = new_length
        self.tokenizer.model_max_length = new_length

    def _set_model_weight(self, model_path=None):
        """
        Initialize model embedding vectors from parameter or from config
        :param model_path:
        :return:
        """
        if model_path is None:
            self.model = AutoModel.from_pretrained(self.config)
        else:
            self.config = AutoConfig.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # load the fine-tuned bert model
            self.model = AutoModel.from_pretrained(model_path)
        #return model

    def featurize(self, sentences: List, with_context=True) -> List:
        """
        Return individual word weight
        :param with_context: contextualized word embeddings
        :param sentences: list of sentences
        :return:
        """
        if not with_context:
            input_ids = [self.tokenizer.encode(s) for s in sentences]
            word_features = [self.vectors[ids].detach().numpy() for ids in input_ids]
        else:
            #sent_pairs = [self.encode_sentence_pair(sentences[i:i+2]) for i in range(len(sentences))]
            #word_features = [self.model(pair[0])['last_hidden_state'].detach().numpy().squeeze() for pair in sent_pairs]
            word_features = self.featurize_tensor(sentences)
        return word_features

    def featurize_tensor(self, sentences: List) -> List:
        """
        :param sentences:
        :return:
        """
        #sent_pairs = [self.tokenizer.sep_token.join(sentences[i:i+2]) for i in range(len(sentences))]
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True)
        inputs_to_model_device = {k:inputs[k][:,:self.config.max_position_embeddings].to(self.model.device) for k in inputs}
        feats_tensor = self.model(**inputs_to_model_device)['last_hidden_state'].cpu().detach().numpy()
        mask_numpy = inputs['attention_mask'].cpu().detach().numpy()
        indices =  [np.where(mask_numpy[i] == 1)[0] for i in range(mask_numpy.shape[0])]
        word_features = [feats[idx] for feats, idx in zip(feats_tensor, indices)]
        return word_features

    def encode_sentence_pair(self, sentence_pair: List) -> Tuple:
        """
        :param sentence_pair: one list of two adjacent sentences
        :return:
        """
        sentences = self.tokenizer.sep_token.join(sentence_pair)
        inputs = self.tokenizer(sentences, return_tensors='pt')
        sep_idx = np.where(inputs['input_ids'].detach().numpy() == self.tokenizer.sep_token_id)[1][0]
        #features = self.model(**inputs)['last_hidden_state'].cpu()
        return (inputs['input_ids'][:,:sep_idx+1], inputs['input_ids'][:,sep_idx:])

    def encode_article(self, sentences: List) -> Tensor:
        """
        Return context features of individual word using bert model
        :param sentences: sentences in one article
        :return:
        """
        #sent_ids = [self.encode_sentence_pairsentences[i:i+2][0] for i in range(len(sentences))]
        sent_ids = [self.tokenizer(sent, return_tensors='pt')['input_ids'] for sent in sentences]
        #features = torch.cat([self.model(ids)['last_hidden_state'] for ids in sent_ids], dim=0)
        input_ids = torch.cat(sent_ids, dim=1)
        return input_ids

    def sentences_mean_repr(self, sentences: List, with_context: bool=True) -> List:
        """
        Get mean word embeddings for each sentences
        :param sentences:
        :return:
        """
        words_features = self.featurize(sentences, with_context)
        sent_features = [np.expand_dims(np.mean(words, axis=0), axis=0) for words in words_features]
        return sent_features

    def sents_to_text(self, sents) -> str:
        """
        :param sep_token:
        :return:
        """
        return self.tokenizer.sep_token.join(sents)

    def encode_batch(self, texts: list, add_eos: bool = False) -> Tuple:
        """
        :param eos_tensor: eos tensor must be given if the tokenizer doesn't encode eos token automatically
        :param texts: each text contains list of sentences
        :return: sorted batch : input ids, attention mask, sequence lens
                 original indices
        """

        # Encode the sentences in each text and sort the sentences according to the length
        text_lens = {}
        ids = []
        for i, text in enumerate(texts):
            #t_tensor, id_tensor = self.featurize_article(text)
            id_tensor = self.encode_article(text)
            if add_eos:
                #t_tensor = torch.cat((t_tensor, eos_tensor),dim = 0)
                id_tensor = torch.cat((id_tensor, torch.tensor(self.tokenizer.eos_token_id).long()), dim = 1)
            ids.append(id_tensor)
            text_lens[i] = id_tensor.shape[1]

        #print('len of text lens original dict ', text_lens)
        sorted_lens = {k: v for k, v in sorted(text_lens.items(),
                                               key=lambda item: item[1], reverse=True)}
        #print('sorted lengths encoded ', sorted_lens)
        seq_size = list(sorted_lens.values())[0]
        batch_size = len(texts)
        #input_tensor = torch.zeros((batch_size, seq_size, embed_size))
        mask_tensor = torch.zeros((batch_size, 1, seq_size))
        id_tensor = torch.zeros((batch_size, seq_size))

        for i, cur_idx in enumerate(list(sorted_lens.keys())):
            cur_len = sorted_lens[cur_idx]
            #input_tensor[i][:cur_len] = batch_tensors[cur_idx]
            mask_tensor[i][0][:cur_len] += 1.0
            #print('cur len', cur_len, 'cur_id tensor shape ', ids[cur_idx].shape)
            id_tensor[i][:cur_len] += ids[cur_idx][0]
            if cur_len < seq_size:
                id_tensor[i][cur_len:] += self.tokenizer.pad_token_id

        mask_tensor = mask_tensor == 1.0
        id_tensor = id_tensor.long()
        lens_tensor = torch.from_numpy(np.array(list(sorted_lens.values()))).long()
        batch = {'input_ids':id_tensor, 'attention_mask': mask_tensor}
        return batch, lens_tensor, list(sorted_lens.keys())

    def decode_batch(self, ids_array):
        """
        :param ids_array:
        :return:
        """
        articles = self.tokenizer.decode(ids_array)
        return articles


    def dim_size(self) -> int:
        return self.vectors.size(-1)

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def __repr__(self):
        return "TransformerLexicon(model=%r, tokenizer=%r)" % (
        self.model, self.tokenizer)




    





