

import numpy as np
import torch
import random

def encode_article(tokenizer, sentences, max_positions):
        """
        Return context features of individual word using bert model
        :param sentences: sentences in one article
        :return:
        """
        #sent_ids = [self.encode_sentence_pairsentences[i:i+2][0] for i in range(len(sentences))]
        sent_ids = [tokenizer(sent, return_tensors='pt')['input_ids'] for sent in sentences]
        #features = torch.cat([self.model(ids)['last_hidden_state'] for ids in sent_ids], dim=0)
        input_ids = torch.cat(sent_ids, dim=1)
        return input_ids[:, :max_positions]

def encode_batch(batch, pad_token_id, eos_token_id = None):
        """
        :param eos_tensor: eos tensor must be given if the tokenizer doesn't encode eos token automatically
        :param texts: each text contains list of sentences
        :return: sorted batch : input ids, attention mask, sequence lens
                 original indices
        """

        # Encode the sentences in each text and sort the sentences according to the length
        text_lens = {}
        ids = []
        for i, id_tensor in enumerate(batch):
            #id_tensor = encode_article(tokenizer,text)
            #id_tensor = encode_article(tokenizer, text)
            if eos_token_id is not None:
                #t_tensor = torch.cat((t_tensor, eos_tensor),dim = 0)
                id_tensor = torch.cat((id_tensor, torch.tensor(eos_token_id).long()), dim = 1)
            ids.append(id_tensor)
            text_lens[i] = id_tensor.shape[1]

        #print('len of text lens original dict ', text_lens)
        sorted_lens = {k: v for k, v in sorted(text_lens.items(),
                                               key=lambda item: item[1], reverse=True)}
        #print('sorted lengths encoded ', sorted_lens)
        seq_size = list(sorted_lens.values())[0]
        batch_size = len(batch)
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
                id_tensor[i][cur_len:] += pad_token_id

        mask_tensor = mask_tensor == 1.0
        id_tensor = id_tensor.long()
        lens_tensor = torch.from_numpy(np.array(list(sorted_lens.values()))).long()
        batch = {'input_ids':id_tensor, 'attention_mask': mask_tensor}
        return batch, lens_tensor, list(sorted_lens.keys())


class SummaryData(torch.utils.data.Dataset):
    """Construct data set for dataloader"""
    def __init__(self, dataset, sep_token, eos_token = None, ground_truth = True, select_random= False):
        """
        # A sep token is important for indicating the structure of the data
        """
        self.longest = []
        for text in dataset:
            if not select_random:
                k = 6 - len(text[0][:2])
                top_k_sents = {i:len(sent) for i, sent in enumerate(text[1])}
                sorted_topk = sorted(top_k_sents.items(), key=lambda item: item[1], reverse=True)[:k]
                indices = sorted([p[0] for p in sorted_topk])
                longest = sep_token.join(text[0][:2] + [text[1][i] for i in indices])
            else:
                indices = [i for i in range(len(text[1]))]
                random.shuffle(indices)
                longest = sep_token.join([text[1][i] for i in indices[:5]])
            
            self.longest.append(longest)

        self.sources = [sep_token.join(sent[0][:2] + sent[1]) for sent in dataset]
        if ground_truth:
            if eos_token is not None:
                self.targets = [sep_token.join(sent[2]) + eos_token for sent in dataset]
                
            else:
                self.targets = [sep_token.join(sent[2]) for sent in dataset]

            #self.targets = [bos_token + seq for seq in join_targets]

        else:
            self.targets = None

    def __getitem__(self, index):
        """
        :param index : int
        """
        src = self.sources[index]
        longest = self.longest[index]
        if self.targets is not None:
            trg = self.targets[index]
            return {'src': src, 'trg': trg, 'longest':longest}
        else:
            return {'src':src, 'longest':longest}

    def __len__(self):
        return len(self.sources)