from logging import log, logProcesses
from typing import List, Dict, Tuple, Callable, Any, Optional, Union, Iterable
import os
import numpy as np
from numpy import ndarray
import torch
from torch import nn
from torch import Tensor
from torch._C import NoneType
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.file_utils import add_code_sample_docstrings
from pointer_model.search import beam_search

import pytorch_lightning as pl
from joeynmt.helpers import make_model_dir, load_config, symlink_update, tile, load_checkpoint
from joeynmt.loss import XentLoss
from joeynmt.builders import build_scheduler, build_gradient_clipper, build_optimizer
from transformers import (AdamW, WEIGHTS_NAME, get_linear_schedule_with_warmup)
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM, EncoderDecoderModel, AutoConfig, EncoderDecoderConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.file_utils import ModelOutput
from transformers.generation_utils import SampleEncoderDecoderOutput, SampleDecoderOnlyOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput, BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput
from transformers.generation_beam_search import BeamSearchScorer, BeamScorer

from pytorch_lightning.plugins import DeepSpeedPlugin
import deepspeed

class EncoderDecoderModelAddPointer(EncoderDecoderModel):
    """
    """
    def __init__(self,config, encoder = None, decoder = None, vocab_weight = None):
        super().__init__(config, encoder, decoder)

        self.vocab_weight = vocab_weight
        self._init_pointer()
        #self._set_loss_func(loss_func, pad_token_id)
        
        #self.config.is_encoder_decoder = True

    def _init_pointer(self):
        """
        Pointer input contains target embed, target hidden output, source hidden output
        :param input_size (target embed size + target hidden size + source hidden size)

        """
        input_size = self.decoder.config.hidden_size * 3
        self.pointer = nn.Linear(input_size, 1, bias=True)

    def _set_loss_func(self, loss_func: Callable, pad_token_id: int):
        """
        Use NLLoss that takes log_softmax output as input instead of logits
        :param loss_func:
        :param pad_token_id:
        :return:
        """
        if loss_func is None:
            #loss_func = XentLoss(pad_token_id)
            if self.vocab_weight is not None:
                print('NLLoss with vocab weight ', self.vocab_weight.shape)
                loss_func = nn.NLLLoss(weight=self.vocab_weight.to(self.decoder.device), ignore_index = pad_token_id)
            else:
                loss_func = nn.NLLLoss(ignore_index = pad_token_id)
            
                
            #loss_func = nn.CrossEntropyLoss(ignore_index = pad_token_id)
        self.loss_func = loss_func

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
            }
            #input_ids = input_ids.to(encoder.device)
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
        model_kwargs['encoder_input_ids'] = input_ids
        #print('prepared encoder decoder kwargs keys ', model_kwargs.keys())
        return model_kwargs
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past=None, cross_attentions=None, decoder_hidden_states=None, past_logits = None, attention_mask=None, use_cache=None, encoder_outputs=None, encoder_input_ids = None, **kwargs):
        # overwrite the original function
        decoder_inputs = self.decoder.prepare_inputs_for_generation(decoder_input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "full_input_ids": decoder_input_ids,
            "past_logits": past_logits, 
            "past_key_values": decoder_inputs['past_key_values'],
            "past_cross_attentions": cross_attentions,
            "past_decoder_hidden_states": decoder_hidden_states,
            "use_cache": use_cache,
            'input_ids': encoder_input_ids,
        }
        return input_dict

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past, overwrite the original function
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        #assert 'cross_attentions' in outputs, 'cross attentions must be returned in outputs'
        
        model_kwargs['cross_attentions'] = outputs.cross_attentions
        model_kwargs['decoder_hidden_states'] = outputs.decoder_hidden_states
        model_kwargs['past_logits'] = outputs.logits

        return model_kwargs
    
    def forward(self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        full_input_ids = None,
        past_logits = None,
        past_cross_attentions=None,
        past_decoder_hidden_states = None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        longest_input_ids = None,
        longest_attention_mask = None,
        labels=None,
        use_cache=None,
        output_attentions=True,
        output_hidden_states=None,
        return_dict=None,
        pointer_ratio=1.0,
        coverage_weight = 0.0,):
        r"""
        Examples::

            >>> from transformers import EncoderDecoderModel, BertTokenizer
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints

            >>> # forward
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

            >>> # training
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
            >>> loss, logits = outputs.loss, outputs.logits

            >>> # save and load from pretrained
            >>> model.save_pretrained("bert2bert")
            >>> model = EncoderDecoderModel.from_pretrained("bert2bert")

            >>> # generation
            >>> generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)

        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_hidden_states = encoder_outputs[0]
        

        # Decode
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        if full_input_ids is None:
            decoder_inputs_embeds = self.decoder.bert.embeddings(decoder_input_ids.to(self.decoder.device), past_key_values_length = past_key_values_length)
        else:
            decoder_inputs_embeds = self.decoder.bert.embeddings(full_input_ids.to(self.decoder.device))
        
        if labels is not None:
            labels = labels.to(self.decoder.device)

        decoder_outputs = self.decoder(
            input_ids =  decoder_input_ids.to(self.decoder.device),
            attention_mask=decoder_attention_mask.to(self.decoder.device),
            encoder_hidden_states=encoder_hidden_states.to(self.decoder.device),
            encoder_attention_mask=attention_mask.to(self.decoder.device),
            inputs_embeds=None,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True)

        # Reconstruction outputs: the longest sequences from the source output, only for training currently
        if longest_input_ids is not None:
            reconstruct_ouputs = self.decoder(
            input_ids = longest_input_ids.to(self.decoder.device),
            attention_mask = longest_attention_mask.to(self.decoder.device),
            encoder_hidden_states = encoder_hidden_states.to(self.decoder.device),
            encoder_attention_mask = attention_mask.to(self.decoder.device), labels=longest_input_ids.to(self.decoder.device), return_dict=True)
        else:
            reconstruct_ouputs = None

        if past_cross_attentions is None:
            past_cross_attentions = decoder_outputs.cross_attentions[-1]
        else:
            past_cross_attentions = torch.cat((past_cross_attentions, decoder_outputs.cross_attentions[-1]), dim=-2)
            
        src_trg_attention = torch.sum(past_cross_attentions, dim = 1)

        #print('src_trg_attention shape', src_trg_attention.shape, )
        if len(src_trg_attention.size()) > 3:
                src_trg_attention = src_trg_attention.squeeze(1)
            #print('src_trg_atteion shape', src_trg_attention.shape)
        
        #encoder_hidden_states = encoder_hidden_states.type_as(src_trg_attention)
        encoder_hiddens_attended = torch.matmul(src_trg_attention, encoder_hidden_states)

        if past_decoder_hidden_states is None:
            decoder_hidden_last = decoder_outputs.hidden_states[-1]
        else:
           decoder_hidden_last = torch.cat((past_decoder_hidden_states, decoder_outputs.hidden_states[-1]), dim = -2)

        
        pointer_inputs = torch.cat((decoder_inputs_embeds, encoder_hiddens_attended, decoder_hidden_last), dim=-1)
        
        p_gen = torch.sigmoid(self.pointer(pointer_inputs))
        
        if past_logits is not None:
            past_logits = torch.cat((past_logits, decoder_outputs.logits), dim = -2)
        else:
            past_logits = decoder_outputs.logits

       
        ## Pointer on logits
        #prediction_scores = p_gen * decoder_outputs.logits + (1-p_gen) * self.get_output_embeddings()(encoder_hiddens_attended)
        #prediction_scores = p_gen * past_logits + (1-p_gen) * self.get_output_embeddings()(encoder_hiddens_attended)

        ## Pointer on softmax between decoder and encoder hiddens logits
        #prediction_scores = p_gen * F.log_softmax(past_logits, dim=-1) + (1-p_gen) * F.log_softmax(self.get_output_embeddings()(encoder_hiddens_attended), dim=-1)
        #prediction_scores = p_gen * F.softmax(past_logits, dim=-1) + (1-p_gen) * F.softmax(self.get_output_embeddings()(encoder_hiddens_attended), dim=-1)
        
        ## Add more values 
        #past_logits += self.get_output_embeddings()(encoder_hiddens_attended), dim=-1)
        ## Pointer on softmax between decoder logits and src trg attentions 
        #past_scores = F.softmax(past_logits, dim=-1)
        #self.past_scores = past_scores.detach().cpu()
        #past_scores *= p_gen
        past_scores = p_gen * F.softmax(past_logits, dim=-1)
        src_trg_attention_mean = torch.mean(past_cross_attentions, dim = 1)
        self.cross_attention = src_trg_attention_mean.detach().cpu()

        if len(src_trg_attention_mean.size()) > 3:
                src_trg_attention_mean = src_trg_attention_mean.squeeze()
        src_trg_attention_mean *= (1-p_gen)

        # Fix ids dimension 
        if len(input_ids.size()) < 3:
                input_ids = input_ids.unsqueeze(1)
        src_ids = tile(input_ids, src_trg_attention_mean.size(1), dim=1).to(self.decoder.device)
        
        prediction_scores = past_scores.scatter_add_(dim=2, index=src_ids, src=src_trg_attention_mean)
        
       
        # Customized loss func is NLLoss that takes log_softmax output as input
        # Avoid -inf and nan loss
        prediction_scores = torch.log(prediction_scores+0.000003)
        if labels is not None:
                labels = labels.type_as(prediction_scores).long()
              
                # Coverage
                #coverage = torch.empty_like(src_trg_attention).copy_(src_trg_attention_mean)
                #if coverage_weight > 0.:
                #    for t in range(1,prediction_scores.size(1)):
                #        sum_t = torch.sum(src_trg_attention[:,0:t],dim=1)
                #        out = torch.where(sum_t < coverage[:, t], sum_t, coverage[:,t])
                #        coverage[:,t] = out
                #    coverage = coverage.sum(-1).unsqueeze(2)
        
                #coverage = tile(coverage, prediction_scores.size(2), dim=2)
                #shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous() - coverage_weight * coverage[:, 1:, :].contiguous()
                shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
                labels = labels[:, 1:].contiguous()
                loss_p = self.loss_func(shifted_prediction_scores.view(-1, prediction_scores.shape[-1]), labels.view(-1))
                #pointer_ratio = self.model_config['pointer_ratio'] # pointer_ratio = 0.5
                if reconstruct_ouputs is not  None:
                    #decoder_outputs.loss = pointer_ratio * loss_p + (1-pointer_ratio) * reconstruct_ouputs.loss
                    decoder_outputs.loss = loss_p + reconstruct_ouputs.loss
                else:
                    #loss = pointer_ratio * loss_p + (1-pointer_ratio) * decoder_outputs.loss
                    decoder_outputs.loss = loss_p
        
        self.p_gen = p_gen.detach().cpu().numpy().tolist()

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=prediction_scores,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_hidden_last,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,)


class Bert2BertModel(pl.LightningModule):
    """
    EncoderDecoderModel + pointer based on pre-trained BERT models
    """
    def __init__(self, config_file, loss_func=None, reconstruct=False, pointer_ratio = 0., coverage_weight= 0.,vocab_weight= None):
        """
        Initialize EncoderDecoderModel with the configs from the bert models defined in the config file.
        :param config_file:
        :param loss_func: 
        """
        super().__init__()
        self._load_configs(config_file)
        self.save_hyperparameters()
        
        encoder_name = self.model_config['encoder']['name']
        encoder_max_position = self.model_config['encoder']['max_position']
        encoder_config = AutoConfig.from_pretrained(encoder_name)
        self.enc_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        if 'bert' in encoder_name:
            encoder_config.max_position_embeddings = encoder_max_position
            self.enc_pad_token_id = self.model_config['encoder']['pad_token_id']
            
        else:
            encoder_config.n_positions = encoder_max_position
            self.enc_pad_token_id = self.enc_tokenizer.eos_token_id
       
        self.enc_tokenizer.model_max_length = encoder_max_position


        decoder_name = self.model_config['decoder']['name']
        decoder_max_position = self.model_config['decoder']['max_position']
        decoder_config = AutoConfig.from_pretrained(decoder_name, is_decoder=True, add_cross_attention=True)
        self.dec_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        if 'bert' in decoder_name:
            decoder_config.max_position_embeddings = decoder_max_position
            self.dec_pad_token_id = self.model_config['decoder']['pad_token_id']
        
        else:
            decoder_config.n_positions = decoder_max_position
            self.dec_pad_token_id = self.dec_tokenizer.eos_token_id
            
        self.dec_tokenizer.model_max_length = decoder_max_position
        self.output_size = self.dec_tokenizer.vocab_size

        enc_dec_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config,decoder_config)
        
        if pointer_ratio == 0.:
            self.add_pointer = False
            print('no pointer')
        else:
            self.add_pointer = True
            self.pointer_ratio = pointer_ratio
            print('pointer ratio ', pointer_ratio)
        
        self.coverage_weight = coverage_weight

        self.reconstruct = reconstruct

        self.vocab_weight = vocab_weight

        if not self.add_pointer: 
            self.model = EncoderDecoderModel(config=enc_dec_config)
        else:
            self.model = EncoderDecoderModelAddPointer(config=enc_dec_config,  vocab_weight=self.vocab_weight)
            self.model._set_loss_func(loss_func=loss_func, pad_token_id=self.dec_pad_token_id)
       
        print('self.model devices ', self.model.device, self.model.encoder.device, self.model.decoder.device)

        self.second_decoder = None
        # Use masked lm for refining the generation from the first stage
        if 'second_decoder' in self.model_config:
            self.second_decoder = AutoModelForMaskedLM.from_pretrained(self.model_config['second_decoder']['name'])

    
    def _get_input_embed(self, encoder:bool = True):
        """
        Returns the embedding module of either encoder or decoder
        :param encoder:
        :return:
        """
        if encoder:
            embed = self.model.get_input_embeddings()
        else:
            embed = self.model.decoder.get_input_embeddings()
        return embed

    def _get_output_embed(self):
        """
        The embedding module of the decoder
        :return:
        """
        return self.model.get_output_embeddings()

    def _model_special_tokens(self, bos_id=None, eos_id=None, pad_id = None):
        """
        """
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id
        self.pad_token_id = pad_id 
    

    def _load_pretrained_weights(self, encoder_name: str = None, decoder_name:str = None, enc_dec_name: str = None):
        """
        Load pre-trained bert models to the encoder and decoder except position embeddings and cross attention modules
        :param encoder_name: pre-trained model path
        :param decoder_name: pre-trained model path
        :param enc_dec_name: pre-trained model path
        """
        if enc_dec_name is not None:
            if not self.add_pointer:
                self.model = EncoderDecoderModel.from_pretrained(enc_dec_name)
            else:
                self.model = EncoderDecoderModelAddPointer.from_pretrained(enc_dec_name)
        
        else:
            encoder_model = AutoModel.from_pretrained(encoder_name)
            # BertLMHeadModel or GPT2LMHeadModel
            decoder_model = AutoModelForCausalLM.from_pretrained(decoder_name, is_decoder=True)
            bert_model_name_paras = {para[0]:para[1].detach().clone() for para in encoder_model.named_parameters()}
            bert_causal_model_paras = {para[0]:para[1].detach().clone() for para in decoder_model.named_parameters()}
            
            # Load weights except position embeddings
            for paras in self.model.encoder.named_parameters():
                if  'embeddings.position_embeddings.weight' not in paras[0]:
                    paras[1].data =  bert_model_name_paras[paras[0]]

            for dec_ps in self.model.decoder.named_parameters():
                if 'embeddings.position_embeddings.weight' not in dec_ps[0] and "crossattention" not in dec_ps[0]:
                        dec_ps[1].data = bert_causal_model_paras[dec_ps[0]]

            del encoder_model, decoder_model, bert_model_name_paras, bert_causal_model_paras

    
    def _load_configs(self, config_file):
        """
        Load the data, model, training, testing configs from one yaml file
        :param config_file: 
        :return: 
        """
        config = load_config(config_file)
        self.data_config = config['data']
        self.model_config = config['model']
        self.train_config = config['train']
        self.test_config = config['test']
        self.model_dir = self.train_config['model_dir']

    def _encode(self, inputs_embeds = None, input_ids = None, attention_mask = None):
        """
        Encoder encodes batch of source input, sorted input
        :param inputs_embeds: output from embeddings module
        :param input_ids: tokenized ids
        :param attention_mask: (batch, seq)
        :return: output: hidden states

        """
        return self.model.encoder(inputs_embeds = inputs_embeds, input_ids = input_ids, attention_mask = attention_mask)

    
    def _decode_second_stage(self,  decoder_inputs_embeds=None, 
                    decoder_attention_mask=None, 
                    encoder_hidden_states=None,
                    past_key_values=None,   
                    labels=None, use_cache=None, 
                    output_attentions=None, 
                    output_hidden_states=None):
        """
        Second decoder for refining the generation.
        :param decoder_inputs_embeds:
        :param decoder_attention_mask:
        :param encoder_hidden_states:
        :param past_key_values:
        :param labels: target ids
        :param use_cache: True
        :param output_attentions: False
        :param output_hidden_states: False
        :return:
        """

        return self.second_decoder(inputs_embeds=decoder_inputs_embeds, 
                                            attention_mask=decoder_attention_mask, 
                                            encoder_hidden_states=encoder_hidden_states, 
                                            past_key_values=past_key_values,
                                            labels=labels, use_cache=use_cache, 
                                            output_attentions=output_attentions, 
                                            output_hidden_states=output_hidden_states, return_dict=True)

    
    def forward(self, input_ids = None, inputs_embeds = None, attention_mask = None, 
                    decoder_input_ids = None, decoder_attention_mask=None,
                    encoder_outputs=None, past_key_values=None, longest_input_ids = None, longest_attention_mask = None,
                    labels = None, use_cache=None, inference = True):
        """

        :param input_ids: tokenized ids of source text
        :param inputs_embeds: output from the embeddings of encoder
        :param attention_mask: ignore the pad token ids in encoder attention
        :param decoder_inputs_embeds: output from the embeddings of decoder
        :param decoder_attention_mask: ignore the pad token ids in decoder attetion
        :param encoder_hidden_states: encoder hidden states (batch size, sequence length, hidden size)
        :param past_key_values: 
        :param mask_logits: mask token embeddings for masking prediction of low scores that will be regenerated by the seconde decoder
        :param labels: target ids
        :param use_cache:
        :param output_attentions: true if add pointer
        :param output_hidden_states: true if add pointer
        :return:
        """
        print('self.model devices in forward', self.model.device, self.model.encoder.device, self.model.decoder.device)

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(input_ids=input_ids.to(self.model.encoder.device), inputs_embeds=inputs_embeds, attention_mask = attention_mask.to(self.model.encoder.device))

        if inference:
            output_attentions = True
        else:
            output_attentions = False
        if self.add_pointer:
            if not self.reconstruct:
                longest_input_ids = None
                #longest_attention_mask = None

            decoder_outputs = self.model( input_ids = input_ids, attention_mask = attention_mask, encoder_outputs = encoder_outputs, decoder_input_ids = decoder_input_ids,decoder_attention_mask = decoder_attention_mask,
                                         longest_input_ids = longest_input_ids, longest_attention_mask = longest_attention_mask, labels = labels, pointer_ratio = self.pointer_ratio, coverage_weight=self.coverage_weight, use_cache=use_cache,
                                            past_key_values = past_key_values, return_dict=True)
            
        else:
            decoder_input_ids = decoder_input_ids.to(self.model.encoder.device)
            decoder_attention_mask = decoder_attention_mask.to(self.model.encoder.device)
            labels = labels.to(self.model.encoder.device)
            decoder_outputs = self.model(input_ids = None, attention_mask = None, encoder_outputs = encoder_outputs, decoder_input_ids = decoder_input_ids,decoder_attention_mask = decoder_attention_mask,
                                         labels = labels, use_cache=use_cache,
                                            past_key_values = past_key_values, 
                                            output_attentions = output_attentions,
                                            return_dict=True)
            if self.reconstruct:
                #if longest_input_ids is not None:
                longest_input_ids = longest_input_ids.to(self.model.encoder.device)
                longest_attention_mask = longest_attention_mask.to(self.model.encoder.device)

                reconstruct_outputs = self.model(input_ids = None, attention_mask = None, encoder_outputs = encoder_outputs, decoder_input_ids = longest_input_ids,decoder_attention_mask = longest_attention_mask,
                                         labels = longest_input_ids, use_cache=use_cache,
                                            past_key_values = past_key_values, output_attentions=output_attentions,
                                            return_dict=True)

                decoder_outputs.loss += reconstruct_outputs.loss

        return decoder_outputs

    def _generate(self, batch, num_beams, do_sample, num_sequences, max_length = None, no_repeat_ngram_size = 0, top_k=50, top_p=0.95, repetition_penalty=1.2):
        """
         return EncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
        """
        src_batch, longest_batch = self._tokenize(batch)
        bos_token_id = self.dec_tokenizer.cls_token_id
        eos_token_id = self.dec_tokenizer.sep_token_id
        pad_token_id = self.dec_tokenizer.pad_token_id

        if max_length is None:
            # either /4 or /5
            #max_length = int(src_batch['input_ids'].size(1) / 4)
            max_length = longest_batch['input_ids'].size(1)
            #max_length = src_batch['input_ids'].size(1)

    
        model_args = {'input_ids': src_batch['input_ids'].to(self.model.encoder.device), 'attention_mask':src_batch['attention_mask'].to(self.model.encoder.device),'use_cache': True, }
        
        outputs = self.model.generate(num_beams=num_beams, num_return_sequences = num_sequences, max_length=max_length, no_repeat_ngram_size = no_repeat_ngram_size, top_k = top_k, top_p = top_p,
                                                bos_token_id = bos_token_id, eos_token_id = eos_token_id, pad_token_id = pad_token_id, do_sample = do_sample, repetition_penalty=repetition_penalty,
                                                early_stopping=True, output_scores=True, return_dict_in_generate = True,**model_args)

        
        if self.add_pointer:
            return self.dec_tokenizer.batch_decode(outputs.sequences), outputs.sequences, outputs.scores, outputs, self.model.p_gen, self.model.cross_attention
        
        return self.dec_tokenizer.batch_decode(outputs.sequences), outputs.sequences, outputs.scores, outputs, None, None


    def _tokenize(self, batch):
        """
        :param batch: dict
        """
        
        xs = batch['src']
        src_batch = self.enc_tokenizer(xs, return_token_type_ids=False, return_tensors='pt', padding=True)
        src_batch = {k:v[:,:self.enc_tokenizer.model_max_length] for k, v in src_batch.items()}

        longest = batch['longest']
        src_longest_batch = self.enc_tokenizer(longest, return_token_type_ids=False, return_tensors='pt', padding=True)
        src_longest_batch = {k:v[:,:self.enc_tokenizer.model_max_length] for k, v in src_longest_batch.items()}

        if 'trg' in batch: 
            ys = batch['trg']
            trg_batch = self.dec_tokenizer(ys, return_token_type_ids=False, return_tensors='pt', padding=True)
            trg_batch = {k:v[:,:self.dec_tokenizer.model_max_length] for k, v in trg_batch.items()}
            return src_batch, src_longest_batch, trg_batch
        else:
            return src_batch, src_longest_batch


    def training_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        #print(len(batch))
        src_batch, longest_batch, trg_batch = self._tokenize(batch)
        # decoder_inputs_embeds for pointer input
        #decoder_inputs_embeds = self.decoder_embed(trg_batch['input_ids'])
        outputs = self(input_ids = src_batch['input_ids'], attention_mask = src_batch['attention_mask'], decoder_input_ids = trg_batch['input_ids'], decoder_attention_mask = trg_batch['attention_mask'],
                                labels = trg_batch['input_ids'], longest_input_ids=longest_batch['input_ids'], longest_attention_mask = longest_batch['attention_mask'])

        return outputs.loss

    def validation_step(self, batch, batch_idx):

        src_batch, longest_batch, trg_batch = self._tokenize(batch)
        outputs = self(input_ids = src_batch['input_ids'], attention_mask = src_batch['attention_mask'], decoder_input_ids = trg_batch['input_ids'], decoder_attention_mask = trg_batch['attention_mask'],
                                labels = trg_batch['input_ids'],longest_input_ids=longest_batch['input_ids'], longest_attention_mask = longest_batch['attention_mask'])

        self.log('val_loss', outputs.loss)
        

    def test_step(self, batch, batch_idx):
        
        src_batch, longest_batch, trg_batch = self._tokenize(batch)

        outputs = self(input_ids = src_batch['input_ids'], attention_mask = src_batch['attention_mask'], decoder_input_ids = trg_batch['input_ids'], decoder_attention_mask = trg_batch['attention_mask'],
                                labels = trg_batch['input_ids'],longest_input_ids=longest_batch['input_ids'], longest_attention_mask = longest_batch['attention_mask'])

        self.log('test_loss', outputs.loss)
       
    def configure_optimizers(self):
        """
        :param config: training configs
        :return:
        """
        weight_decay = self.train_config.get("weight_decay", 0.3)
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
                {"params": [p for n, p in self.model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,},
                {"params": [p for n, p in self.model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,}, 
                {"params": [p for n, p in self.model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,},
                {"params": [p for n, p in self.model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,}]


        if self.add_pointer:
            optimizer_grouped_parameters += [
                {"params": [p for n, p in self.model.pointer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,},
                {"params": [p for n, p in self.model.pointer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,}]
        
        
        lr = float(self.train_config['learning_rate'])
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        #print('optimizer lr ', optimizer.param_groups[0]['lr'])
        
        return optimizer

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings
        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s )"% (self.__class__.__name__, self.model.encoder,
                                  self.model.decoder)

