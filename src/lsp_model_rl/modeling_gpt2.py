# coding=utf-8
"""PyTorch OpenAI GPT-2 model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import copy
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Iterable, Optional, Tuple
from torch import Tensor
import time
import nltk
import numpy as np

from transformers import GPT2PreTrainedModel, GPT2Model

from pytorch_pretrained_bert.modeling_gpt2 import GPT2LMHead, Attention, Block, \
	LayerNorm, MLP

# from generate import top_filtering

from .rewards import calc_rewards

# from .generation_utils import GenerationMixin


logger = logging.getLogger(__name__)

import sys, os
sys.path.append("../")
sys.path.append(".../")
import MisinfoCorrect.src.variables_ext as cfg

class AttentionFP16(Attention):
	def __init__(self, nx, n_ctx, config, scale=False):
		super(AttentionFP16, self).__init__(nx, n_ctx, config, scale)

	def _attn(self, q, k, v):
		w = torch.matmul(q, k)
		if self.scale:
			w = w / math.sqrt(v.size(-1))
		nd, ns = w.size(-2), w.size(-1)
		b = self.bias[:, :, ns-nd:ns, :ns]
		w = w * b - 1e4 * (1 - b)    # point out by Yen-Chun, FP16 overflow

		w = nn.Softmax(dim=-1)(w)
		return torch.matmul(w, v)


class BlockFP16(Block):
	def __init__(self, n_ctx, config, scale=False):
		super(BlockFP16, self).__init__(n_ctx, config, scale)
		nx = config.n_embd
		self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
		self.attn = AttentionFP16(nx, n_ctx, config, scale)
		self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
		self.mlp = MLP(4 * nx, config)


class GPT2ModelFP16(GPT2Model):
	def __init__(self, config):
		# super(GPT2ModelFP16, self).__init__(config)
		super().__init__(config)
		self.wte = nn.Embedding(config.vocab_size, config.n_embd)
		self.wpe = nn.Embedding(config.n_positions, config.n_embd)
		block = BlockFP16(config.n_ctx, config, scale=True)
		self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
		self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

		self.init_weights()

class GPT2LMHeadModel(GPT2PreTrainedModel):
	def __init__(self, config):
		super(GPT2LMHeadModel, self).__init__(config)
		self.transformer = GPT2Model(config)
		# lm_head generated hidden state to lm_logits -> vocabulary distribution
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # GPT2LMHead(self.transformer.wte.weight, config)
		self.position_num_labels = 2 # insert/replace: only two
		self.lambda_position = 0.1
		# position classifier as mentioned in the paper
		# actually, the head is a linear layer for the classification
		self.position_classifier = GPT2ClassificationHead(num_labels = self.position_num_labels) #GPT2LMHead(self.transformer.wte.weight, config)
		self.init_weights()

	def set_tied(self):
		""" Make sure we are sharing the embeddings
		"""
		self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

	def get_output_embeddings(self):
		return self.lm_head
	
	def padding_tensor_3D(self, sequences, max_len):
		"""
		:param sequences: list of tensors
		:return:
		"""
		num = len(sequences)
		out_dims = (num, max_len, *sequences[0].shape[1:])
		out_tensor = sequences[0].data.new(*out_dims).fill_(0)

		# print('out_tensor:', out_tensor.shape)

		mask = sequences[0].data.new(*out_dims).fill_(0)
		for i, tensor in enumerate(sequences):
			length = tensor.size(0)
			# print('length:', length)
			out_tensor[i, :length] = tensor
			mask[i, :length] = 1
		return out_tensor, mask

	def padding_tensor_2D(self, sequences, max_len):
		"""
		:param sequences: list of tensors
		:return:
		"""
		num = len(sequences)
		out_dims = (num, max_len)
		out_tensor = sequences[0].data.new(*out_dims).fill_(0)

		# print('out_tensor:', out_tensor.shape)

		mask = sequences[0].data.new(*out_dims).fill_(0)
		for i, tensor in enumerate(sequences):
			length = min(tensor.size(0), max_len)
			# print('length:', length)
			out_tensor[i, :length] = tensor[:length]
			mask[i, :length] = 1
		return out_tensor, mask

	def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, position_labels=None, past=None, seeker_post=None, response_post=None, top_k=60, top_p=0.92, temperature=0.9, eos=None, tokenizer=None, baseline_val=0):

		transformer_start_time = time.time()

		# Forward Transformer Pass
		# self.transformer is a GPT model: no generation at this moment
		hidden_states, presents = self.transformer(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, past=past) 
# 		res = self.transformer(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, past_key_values=past) # past_key_values from past # by ext
# 		hidden_states, presents = res.last_hidden_state, None # by ext last_hidden_state
        
		transformer_end_time = time.time()

		# Get LM and position logits
		lm_logits = self.lm_head(hidden_states) #

		if tokenizer is None:
			return lm_logits, presents

		# ext: i think X2 to expand it to 2k+1: even if we just k positions, we double it.
		# -1: we select the final dimension data: --- dimension-driven understanding
		# like a.shape = 2,3,4 -> :,-1,:->2,4
		position_logits = self.position_classifier(hidden_states[:, -1, :]) # X2: shape

		# A1 (Selecting a position)
		probs_position = torch.softmax(position_logits.view(-1, self.position_num_labels), -1) # (batch_size, num_position)
		all_positions = torch.argmax(probs_position, 1)
		all_positions = all_positions.squeeze()

		all_positions = all_positions.cpu().numpy().tolist()

		# A2 (Candidate Sentence): Sample from DialoGPT
		all_outputs = []
		all_output_ids = []
		all_output_logits = []
		sample_dialogpt_start_time = time.time()

		for ii, _ in enumerate(input_ids):
			curr_seeker = tokenizer.encode(seeker_post[ii] + tokenizer.eos_token)
			curr_seeker = torch.tensor([curr_seeker,])

			# ==== ext: device control ====
			# ==== ext: device control ====
			curr_seeker = curr_seeker # .to(device) # by ext from cuda for .to(device), use the version2
			# self.generate: is a method of class from transformers.PreTrainedModel
			# here parameters, like top_p, top_k, temperature
			generated_output = self.generate(input_ids = curr_seeker,
											 max_length=1000,
											 pad_token_id=tokenizer.eos_token_id,
											 top_p=0.92,
											 top_k=60,
											 temperature=1,
											 num_return_sequences=1)

			curr_output = tokenizer.decode(generated_output[:, curr_seeker.shape[-1]:][0], skip_special_tokens=True)

			curr_output_ids = generated_output[:, curr_seeker.shape[-1]:][0]
			curr_output_ids = curr_output_ids[:hidden_states.shape[1]]
			curr_position_ids = torch.tensor(range(len(curr_output_ids)), dtype=torch.long) # ext: .to(device) for .to(device), use the version2

			curr_output_logits = lm_logits[ii, range(curr_output_ids.shape[0]), curr_output_ids]

			all_outputs.append(curr_output)
			all_output_ids.append(curr_output_ids)
			all_output_logits.append(curr_output_logits)

		log_softmax = nn.LogSoftmax(1)

		all_output_logits, _ = self.padding_tensor_2D(all_output_logits, hidden_states.shape[1])
		all_output_logits = log_softmax(all_output_logits)

		sample_dialogpt_end_time = time.time()


		# Calculate Reward 
		
		rewritten_response = []

		for idx, _ in enumerate(all_outputs):
			curr_seeker_post = seeker_post[idx]
			curr_response = response_post[idx]
			curr_output = all_outputs[idx]
			curr_position = all_positions[idx]

			curr_response_li = nltk.sent_tokenize(curr_response)

			if curr_position == 0:
				curr_rewritten_response = curr_response

			else:
				curr_rewritten_response_li = curr_response_li[:curr_position] + [curr_output] + curr_response_li[curr_position:]
				curr_rewritten_response = '. '.join(curr_rewritten_response_li)
			
			rewritten_response.append(curr_rewritten_response)

		reward_start_time = time.time()
		# TODO: ext: we rewrite this part
		# ==== partner's ====
		# we change the reward in v2, the follow child class
		# reward = calc_rewards(seeker_post, response_post, rewritten_response, _empathy_change=True, _perplexity=True)
		reward = calc_rewards(seeker_post, response_post, rewritten_response,
							  _politeness=True,
							  _coherence=False, _perplexity=True)
		reward_end_time = time.time()

		batches = np.arange(input_ids.shape[0]).tolist()

		rl_loss = - (reward - baseline_val) * (-torch.mean(all_output_logits[batches,]) + torch.mean(torch.log(probs_position[batches, all_positions]) ))

		return rl_loss, reward

	#  ext: by search on github repo: this function is not used
	def forward_pointwise(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
		hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
		# import pdb; pdb.set_trace()
		lm_logits = self.lm_head(hidden_states)
		if lm_labels is not None:
			# loss_fct = CrossEntropyLoss(ignore_index=-1)
			# loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
			loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction='none')
			loss1 = loss_fct1(lm_logits.view(-1, lm_logits.size(-1)),
							  lm_labels.view(-1))
			loss1 = loss1.view(lm_labels.size(0), lm_labels.size(1))
			label_size = torch.sum(lm_labels != -1, dim=1).type(loss1.type())
			loss1 = torch.sum(loss1, dim=1)/label_size
			ppl1 = torch.exp(loss1)

			return loss1, ppl1
		return lm_logits, presents
	
	def prepare_inputs_for_generation(self, input_ids, **kwargs):
		return {"input_ids": input_ids}

# created by ext for our purpose
class GPT2LMHeadModel_v2(GPT2LMHeadModel):
	def __init__(self, config):
		super(GPT2LMHeadModel_v2, self).__init__(config)
		self.transformer = GPT2Model(config)
		# lm_head generated hidden state to lm_logits -> vocabulary distribution
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # GPT2LMHead(self.transformer.wte.weight, config)
		# self.position_num_labels = 2 # insert/replace: only two
		# self.lambda_position = 0.1 # commented for inheritation: ext
		# TODO: maybe, when we load config, we do not find the weight and it is missing
		# position classifier as mentioned in the paper
		# actually, the head is a linear layer for the classification
		# self.position_classifier = GPT2ClassificationHead(num_labels = self.position_num_labels) #GPT2LMHead(self.transformer.wte.weight, config)
		self.init_weights()

	def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, position_labels=None,
				past=None, seeker_post=None, response_post=None, top_k=60, top_p=0.92, temperature=0.9, eos=None,
				tokenizer=None, baseline_val=0):
		# print("==== use the forward function in version 2 ====")

		transformer_start_time = time.time()

		# Forward Transformer Pass
		# self.transformer is a GPT model: no generation at this moment
		# from the code setup: when we have batches: input_ids can be n-batch-size*length-of-a sentence/input-token-id
		hidden_states, presents = self.transformer(input_ids=input_ids, position_ids=position_ids,
												   token_type_ids=token_type_ids, past=past)
		# 		res = self.transformer(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, past_key_values=past) # past_key_values from past # by ext
		# 		hidden_states, presents = res.last_hidden_state, None # by ext last_hidden_state

		transformer_end_time = time.time()

		# Get LM and position logits
		lm_logits = self.lm_head(hidden_states)  #

		if tokenizer is None:
			return lm_logits, presents

		# ext: i think X2 to expand it to 2k+1: even if we just k positions, we double it.
		# -1: we select the final dimension data: --- dimension-driven understanding
		# like a.shape = 2,3,4 -> :,-1,:->2,4
		# position_logits = self.position_classifier(hidden_states[:, -1, :])  # X2: shape

		# A1 (Selecting a position)
		# probs_position = torch.softmax(position_logits.view(-1, self.position_num_labels),-1)  # (batch_size, num_position)
		# all_positions = torch.argmax(probs_position, 1)
		# all_positions = all_positions.squeeze()

		# all_positions = all_positions.cpu().numpy().tolist()

		# A2 (Candidate Sentence): Sample from DialoGPT
		all_outputs = []
		all_output_ids = []
		all_output_logits = []
		sample_dialogpt_start_time = time.time()

		for ii, _ in enumerate(input_ids):
			curr_seeker = tokenizer.encode(seeker_post[ii] + tokenizer.eos_token)
			curr_seeker = torch.tensor([curr_seeker, ])

			# ==== ext: device control ====
			import sys, os
			sys.path.append("../")
			sys.path.append(".../")
			from MisinfoCorrect.src.variables_ext import device
			# ==== ext: device control ====
			curr_seeker = curr_seeker.to(device)  # by ext from cuda
			# self.generate: is a method of class from transformers.PreTrainedModel
			# here parameters, like top_p, top_k, temperature
			generated_output = self.generate(input_ids=curr_seeker,
											 max_length=140, # by ext from 1000 to 140
											 pad_token_id=tokenizer.eos_token_id,
											 top_p=0.92,
											 top_k=60,
											 temperature=1,
											 num_return_sequences=1)

			curr_output = tokenizer.decode(generated_output[:, curr_seeker.shape[-1]:][0], skip_special_tokens=True)

			curr_output_ids = generated_output[:, curr_seeker.shape[-1]:][0]
			curr_output_ids = curr_output_ids[:hidden_states.shape[1]] # ext: TODO: expanded usage?
			# TODO: or nonsense due to the large number of hidden state
			curr_position_ids = torch.tensor(range(len(curr_output_ids)), dtype=torch.long).to(
				device)  # by ext

			# ext: here, we propability of each vob in the vocabulary, like p1*p2*p3
			curr_output_logits = lm_logits[ii, range(curr_output_ids.shape[0]), curr_output_ids]

			all_outputs.append(curr_output)
			all_output_ids.append(curr_output_ids)
			all_output_logits.append(curr_output_logits)

		log_softmax = nn.LogSoftmax(1)

		all_output_logits, _ = self.padding_tensor_2D(all_output_logits, hidden_states.shape[1])
		all_output_logits = log_softmax(all_output_logits)

		sample_dialogpt_end_time = time.time()

		# Calculate Reward

		rewritten_response = []

		for idx, _ in enumerate(all_outputs):
			curr_seeker_post = seeker_post[idx]
			curr_response = response_post[idx]
			curr_output = all_outputs[idx]
			# curr_position = all_positions[idx]

			curr_response_li = nltk.sent_tokenize(curr_response)

			# ==== previous partner ====
			# if curr_position == 0:
			# 	curr_rewritten_response = curr_response
			# else:
			# 	curr_rewritten_response_li = curr_response_li[:curr_position] + [curr_output] + curr_response_li[
			# 																					curr_position:]
			# 	curr_rewritten_response = '. '.join(curr_rewritten_response_li)
			# ==== version 2 ====

			curr_rewritten_response_li = [curr_output]
			curr_rewritten_response = '. '.join(curr_rewritten_response_li)

			rewritten_response.append(curr_rewritten_response)

		reward_start_time = time.time()
		# TODO: ext: we rewrite this part
		# ==== partner's ====
		# reward = calc_rewards(seeker_post, response_post, rewritten_response, _empathy_change=True, _perplexity=True)
		# ==== by ext ====
		# ======== debug =======: seeker_post: list, ['i like it', 'i forget it']
		# print(f'the current seeker post is: {seeker_post}')
		# print(f'the current rewritten_response is: {rewritten_response}')
		# exit(0)

		if_cut_responses_with_more_characters = True
		if if_cut_responses_with_more_characters:
			rewritten_response_new = []
			for one_response in rewritten_response:
				if len(one_response) > 280:
					rewritten_response_new.append(one_response[:280])					
				else:
					rewritten_response_new.append(one_response)
					
			rewritten_response = rewritten_response_new

		reward = calc_rewards(seeker_post, None, rewritten_response,
							  _politeness=cfg.if_politeness,
							  _refutation=cfg.if_refutation,
							  _evidence=cfg.if_evidence,
							  _perplexity=cfg.if_perplexity,
							  _relevance=cfg.if_relevance)
		reward_end_time = time.time()

		batches = np.arange(input_ids.shape[0]).tolist()

		# rl_loss = - (reward - baseline_val) * (-torch.mean(all_output_logits[batches,]) + torch.mean(
		# 	torch.log(probs_position[batches, all_positions])))
		# ==== version 2 ====
		# log addition -> 0 for probability 1
		# in the release code, baseline_val = 0
		# reward is positive: like cross-entropy setup
		# here, -1 to change the negative value to the positive value
		# some data examples: reward: 0.50, rl_loss is -95, then prob after log: ~200
		# please note: the baseline_val is 0 if we do not initiate it
		rl_loss = - (reward - baseline_val) * (-torch.mean(all_output_logits[batches,]))
		# print(f"the reward, baseline_val, all_output_logits[batches,], rl_loss is:"
		# 	  f"\n {reward}\n{baseline_val},\n{all_output_logits[batches,]},\n{rl_loss}")
		# exit
		return rl_loss, reward

class GPT2ClassificationHead(nn.Module):
	"""Head for sentence-level classification tasks."""

	def __init__(self, hidden_dropout_prob=0.1, hidden_size=1024, num_labels=2):
		super().__init__()

		# self.dense = nn.Linear(hidden_size, hidden_size) # previous by the
		self.dense = nn.Linear(768, hidden_size) # by ext from 768*2 to 1024, 
		self.dropout = nn.Dropout(hidden_dropout_prob)
		self.out_proj = nn.Linear(hidden_size, num_labels)

	def forward(self, features, **kwargs):
		print(f"***{features.shape}***")
		x = features[:, :]
		x = self.dropout(x)
		print(f"***{x.shape}***")
		x = self.dense(x)
		x = torch.relu(x)
		x = self.dropout(x)
		x = self.out_proj(x)
		return x
