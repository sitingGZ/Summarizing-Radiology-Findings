#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/
from typing import Dict, List, Tuple, Callable
import os

from torch.autograd.grad_mode import F
import numpy as np
from data import ReportData, make_train_corpus
from pointer_model.Seq2SeqDataset import SummaryData
from pointer_model.EncoderDecoderModels import Bert2BertModel

from torch.utils.data import DataLoader as batch_gen
import torch

from joeynmt.helpers import load_config, set_seed
from pytorch_lightning import Trainer



#use_cuda = config.use_gpu and torch.cuda.is_available()
def main(config_file):
      # Prepare configurations
    configs = load_config(config_file)
    set_seed(seed=configs['train'].get('random_seed', 42))
    model_dir = configs['train']['model_dir']
    batch_size = configs['train']['batch_size']
    encoder_name = configs['model']['encoder']['name']
    decoder_name = configs['model']['decoder']['name']
    #devices = [configs['model']['encoder']['device'], configs['model']['decoder']['device']]
    #gpus = len(set(devices))
    gpus = 1
    max_epochs = configs['train']['epochs']
    # Get training data and pre-trained language model
    tsv_path = configs['data']['tsv']
    data_loader = ReportData(tsv_path)
    train_data, valid_data, test_data = data_loader.get_middle_range_test_data()
    #train_dataset = SummaryData(train_data, sep_token='[CLS]')
    #valid_dataset = SummaryData(valid_data, sep_token='[CLS]')
    #train_loader = batch_gen(train_dataset, batch_size, num_workers=2)
    #valid_loader = batch_gen(valid_dataset, batch_size, num_workers=2)
    #test_dataset = SummaryData(test_data, sep_token= '[CLS]')
    #test_loader = batch_gen(test_dataset, batch_size)

    # use extracted text source
    #test_data = [line.strip() for line in open('extractive_data/tfidf_extractions.txt')]
    
    
    # Initial EncoderDecoderModel with german bert with maximum positions
    #pointer_ratio = configs['model']['pointer_ratio']
    if 'ratio' in model_dir:
        #pointer_ratio = float(model_dir.split('ratio')[-1])
        #print(pointer_ratio)
        pointer_ratio = configs['model']['pointer_ratio']
    else:
        pointer_ratio = 0.

    if 'vocab_weight' in configs['train']:
        vocab_weight = np.array([float(l.strip().split('\t')[-1]) for l in open (configs['train']['vocab_weight'])])
        vocab_weight = torch.FloatTensor(vocab_weight)
    else:
        vocab_weight = None

    # change pointer ratio to be zero
    bert2bert = Bert2BertModel(config_file, reconstruct = False, pointer_ratio=pointer_ratio, vocab_weight=vocab_weight)
    print('Pointer ratio ', pointer_ratio)
    checkpoint_file = [f for f in os.listdir(model_dir) if '.ckpt' in f][-1]
    checkpoint_path = os.path.join(model_dir, checkpoint_file)
    #if "best_checkpoint" in configs['train'] and os.path.exists(configs['train']['best_checkpoint']):
    if os.path.exists(checkpoint_path):
        #bert2bert.load_from_checkpoint(configs['train']['best_checkpoint'])      
        #checkpoints = torch.load(configs['train']['best_checkpoint'])
        checkpoints = torch.load(checkpoint_path)
        bert2bert.load_state_dict(checkpoints['state_dict'])
       
        print('Loaded best checkpoint: {}'.format(checkpoint_path))
    else:
        bert2bert._load_pretrained_weights(encoder_name, decoder_name)
        print('Loaded weights for encoder {} and decoder {}. '.format(encoder_name, decoder_name))

    #print('Devices after Loaded weights for encoder {} and decoder {}. '.format(bert2bert.model.encoder.device, bert2bert.model.decoder.device))
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        bert2bert.cuda()
    print('bert2bert model device: ', bert2bert.model.device, bert2bert.model.encoder.device, bert2bert.model.decoder.device)
    #batch = ([test_dataset[i][0] for i in range(batch_size)], [test_data[i][1] for i in range(batch_size)])
    #trainer = Trainer(gpus=1)
    #trainer.test(bert2bert, test_loader)
    bert2bert._model_special_tokens(bos_id = bert2bert.dec_tokenizer.cls_token_id, 
                                    eos_id = bert2bert.dec_tokenizer.sep_token_id, 
                                    pad_id = bert2bert.dec_tokenizer.pad_token_id)

    num_beams = configs['test']['beam_size']
    do_sample = configs['test']['do_sample']
    #do_sample = False
    num_beams = [1]
    repeat_block = [0, 5]
    batch_size = 1
    for num in num_beams:
      for n_size in repeat_block:
        print('Number of beams and if do sample', num, do_sample)
        #if num > 5:
        #    batch_size = 2
        save_path = os.path.join(model_dir, 'generation_beam_{}_sample_{}_no_repeat_ngram_{}_longer_repetition_penalty_batch_size_{}.txt'.format(num, do_sample,n_size, batch_size))
        with open(save_path, 'w') as file:
            for size in range(0, len(test_data), batch_size):
            #for size in range(0, 6, batch_size):
                batch = {'src':['[CLS]'.join(t[0][:2]+t[1]) for t in test_data[size:size+batch_size]]}
                top_k_sents = [{i:len(sent) for i, sent in enumerate(text[1])} for text in test_data[size:size+batch_size]]
                sorted_topks = [sorted(top_k.items(), key=lambda item: item[1], reverse=True)[:5] for top_k in top_k_sents]
                indices = [sorted([p[0] for p in sorted_topk]) for sorted_topk in sorted_topks]
                longest = []
                for idx, text in zip(indices, test_data[size:size+batch_size]):
                    longest.append('[CLS]'.join(text[0][:2] + [text[1][i] for i in idx]))
                batch.update({'longest':longest})
                #batch = {'src': test_data[size:size+batch_size]}
                #if configs['model']['add_pointer']:
                #    print('generating using beam search for adding pointer. ')
                #    generated, _ = beam_search(bert2bert, num_beams = 5, src_batch = batch, max_output_length=20,
                #    alpha=0.9, n_best = 2)
                #else:
                print('batch ', size, len(batch['src']))
                #if num > 1:
                    #no_repeat_ngram_size = num
                #else:
                    #no_repeat_ngram_size = 0

                generated, indices, scores, outputs, p_gen, cross_attentions = bert2bert._generate(batch, num_beams = num, do_sample = do_sample, num_sequences = 1, max_length = None, no_repeat_ngram_size = n_size )
                print('len generated', len(generated))
                #if p_gen is not None:
                    #print('p_gen shape', p_gen.shape)
                for g in generated:
                    print('len generate', len(g))
                    #print('len longest', len(l))
                    file.write(g + '\n')
                    #file.write('# ' + l +'\n')
                    #if p_gen is not None:
                    #    file.write('p_gen: ' + str(p_gen[i]) +'\n')
                    #if indices is not None:
                    #    file.write('Token ids: ' + str(indices[i]) + '\n')
                    #if scores is not None:
                    #    file.write('Probability top k: ' + str(scores[i]) + '\n')
                    #if topk_ids is not None:
                    #    file.write('Top k ids: ' + str(topk_ids[i]) + '\n')
                    #if cross_attentions is not None:
                        #file.write('Cross attentions: ' + str(cross_attentions[i]) + '\n')

                    #print(g)
                    #print('source', test_data[i][0][:2])
                    #print('source findings', test_data[i][1])
                    #print('reference', test_data[i][2])
        print('Saved to ', save_path)

if __name__ == "__main__":
    #config_file = 'configs/rnn_config.yaml'
    import sys
    print(sys.argv)
    main(sys.argv[1])