import torch
import codecs
import numpy as np

import tqdm
from tqdm import tqdm

import pandas as pd
import re
import csv
import numpy as np

import time

from sklearn.metrics import f1_score

from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

from transformers import AdamW, RobertaConfig, RobertaForSequenceClassification

import datetime
# 
import sys, os
sys.path.append("../")
sys.path.append(".../")
from MisinfoCorrect.src.variables_ext import politeness_clf_fp, refutation_clf_fp, evidence_clf_fp

class CoherenceClassifier():
	def __init__(self, 
			device,
			model_path,
			batch_size = 2):

		self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
		self.batch_size = batch_size
		self.device = device

		self.model = RobertaForSequenceClassification.from_pretrained(
			"roberta-base", # Use the 12-layer BERT model, with an uncased vocab.
			num_labels = 2, # The number of output labels--2 for binary classification.
							# You can increase this for multi-class tasks.   
			output_attentions = False, # Whether the model returns attentions weights.
			output_hidden_states = False, # Whether the model returns all hidden-states.
		)

		# comment data parallel by ext, due to the missing key errors: the possible reason is
		# that the keys are allocated in different gpus? --- be careful of it? TODO
		# OR: another solution, we put nn.DataParallel()after the loading:
		# the only way it can work is that the saving is through nn.DataParallel: interesting
		# self.model = torch.nn.DataParallel(self.model) # TODO: check the details?? by ext
		print(f"the model path is: {model_path}")
		weights = torch.load(model_path, map_location=self.device) # commented by ext
		self.model.load_state_dict(weights) 

		self.model.to(self.device)


	def predict_empathy(self, original_responses, candidate):

		input_ids = []
		attention_masks = []

		for idx, elem in enumerate(original_responses):

			response_sentence = original_responses[idx] + ' </s> ' + candidate

			encoded_dict = self.tokenizer.encode_plus(
								response_sentence,                      # Sentence to encode.
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 64,           # Pad & truncate all sentences.
								pad_to_max_length = True,
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
						)
			
			input_ids.append(encoded_dict['input_ids'])
			attention_masks.append(encoded_dict['attention_mask'])

		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)

		dataset = TensorDataset(input_ids, attention_masks)

		dataloader = DataLoader(
			dataset, # The test samples.
			sampler = SequentialSampler(dataset), # Pull out batches sequentially.
			batch_size = self.batch_size # Evaluate with this batch size.
		)

		self.model.eval()

		for batch in dataloader:
			b_input_ids = batch[0].to(self.device)
			b_input_mask = batch[1].to(self.device)

			with torch.no_grad():
				(logits, ) = self.model(input_ids = b_input_ids, 
														token_type_ids=None,
														attention_mask=b_input_mask,)
# 				res = self.model(input_ids = b_input_ids, 
# 														token_type_ids=None,
# 														attention_mask=b_input_mask,)
# 				logits = res.logits


			logits = logits.detach().cpu().numpy().tolist()
			predictions = np.argmax(logits, axis=1).flatten()

		return (logits, predictions)


class PolitenessClassifier(CoherenceClassifier):
	pass
	# def __init__(self):
	# 	super().__init__()

class RefutationClassifier(CoherenceClassifier):
	def __init__(self, device, model_path, batch_size=2):
		super().__init__(device, model_path, batch_size)

	def predict_empathy(self, original_responses, candidate):


		test_encode = self.tokenizer.batch_encode_plus(list(zip(original_responses, candidate)), 
								padding='max_length', truncation=True, max_length=128, return_tensors='pt', pad_to_max_length=True)

		test_seq = torch.tensor(test_encode['input_ids'])
		test_mask = torch.tensor(test_encode['attention_mask'])
		# test_token = torch.tensor(test_encode['token_type_ids'])


		test_data = TensorDataset(test_seq, test_mask) #, test_token

		dataloader = DataLoader(
			test_data, # The test samples.
			sampler = SequentialSampler(test_data), # Pull out batches sequentially.
			batch_size = self.batch_size # Evaluate with this batch size.
		)

		def model_eval_and_infer(model, prediction_dataloader, device, if_infer=False, if_have_token_types=True):

			model.eval()
			if not if_infer:
				predictions , true_labels = [], []

				for batch in tqdm(prediction_dataloader):
					
					if if_have_token_types:                
						b_input_ids, b_input_mask, b_token_type, b_labels = batch
					else: 
						b_input_ids, b_input_mask, b_labels = batch
					with torch.no_grad():
						
						if if_have_token_types:

							outputs = model(b_input_ids.to(device), token_type_ids=b_token_type.to(device),
											attention_mask=b_input_mask.to(device))
						else:
							outputs = model(b_input_ids.to(device), attention_mask=b_input_mask.to(device))
						
						b_proba = outputs[0]

						proba = b_proba.detach().cpu().numpy()
						label_ids = b_labels.numpy()

						predictions.append(proba)
						true_labels.append(label_ids)

				print('    DONE.')

				flat_predictions = np.concatenate(predictions, axis=0)
				y_pred = np.argmax(flat_predictions, axis=1).flatten()
				y_true = np.concatenate(true_labels, axis=0)


				return y_pred, y_true
			if if_infer:
				predictions  = []

				for batch in prediction_dataloader:

					if if_have_token_types:                
						b_input_ids, b_input_mask, b_token_type = batch
					else: 
						b_input_ids, b_input_mask = batch
					with torch.no_grad():
						if if_have_token_types:
							outputs = model(b_input_ids.to(device), token_type_ids=b_token_type.to(device),
											attention_mask=b_input_mask.to(device))
						else:
							outputs = model(b_input_ids.to(device), attention_mask=b_input_mask.to(device))
						b_proba = outputs[0]

						proba = b_proba.detach().cpu().numpy()
						# label_ids = b_labels.numpy()

						predictions.append(proba)
						# true_labels.append(label_ids)

				

				flat_predictions = np.concatenate(predictions, axis=0)
				y_pred = np.argmax(flat_predictions, axis=1).flatten()
				# y_true = np.concatenate(true_labels, axis=0)

				return y_pred, flat_predictions

		predictions, logits = model_eval_and_infer(self.model, dataloader, device=self.device, if_infer=True, if_have_token_types=False)
		return (logits, predictions)

		# ==== before Oct 2022 ==== in the past
		# input_ids = []
		# attention_masks = []

		# for idx, elem in enumerate(original_responses):

		# 	response_sentence = original_responses[idx] + ' </s> ' + candidate

		# 	encoded_dict = self.tokenizer.encode_plus(
		# 						response_sentence,                      # Sentence to encode.
		# 						add_special_tokens = True, # Add '[CLS]' and '[SEP]'
		# 						max_length = 64,           # Pad & truncate all sentences.
		# 						pad_to_max_length = True,
		# 						return_attention_mask = True,   # Construct attn. masks.
		# 						return_tensors = 'pt',     # Return pytorch tensors.
		# 				)
			
		# 	input_ids.append(encoded_dict['input_ids'])
		# 	attention_masks.append(encoded_dict['attention_mask'])

		# input_ids = torch.cat(input_ids, dim=0)
		# attention_masks = torch.cat(attention_masks, dim=0)

		# dataset = TensorDataset(input_ids, attention_masks)

	# 	dataloader = DataLoader(
	# 		dataset, # The test samples.
	# 		sampler = SequentialSampler(dataset), # Pull out batches sequentially.
	# 		batch_size = self.batch_size # Evaluate with this batch size.
	# 	)

	# 	self.model.eval()

	# 	for batch in dataloader:
	# 		b_input_ids = batch[0].to(self.device)
	# 		b_input_mask = batch[1].to(self.device)

	# 		with torch.no_grad():
	# 			(logits, ) = self.model(input_ids = b_input_ids, 
	# 													token_type_ids=None,
	# 													attention_mask=b_input_mask,)
	# # 				res = self.model(input_ids = b_input_ids, 
	# # 														token_type_ids=None,
	# # 														attention_mask=b_input_mask,)
	# # 				logits = res.logits


	# 		logits = logits.detach().cpu().numpy().tolist()
	# 		predictions = np.argmax(logits, axis=1).flatten()

	# 	return (logits, predictions)


#  inherited from the refutation classifier when considering both the tweet and reply
class EvidenceClassifier(RefutationClassifier):
	pass
	# def __init__(self):
	# 	super().__init__()


'''
Example:
'''

import sys, os
sys.path.append("./")
sys.path.append("../")
sys.path.append(".../")
sys.path.append("..../")
from MisinfoCorrect.src.variables_ext import device


original_responses = [ 'I am so sorry that she is not getting it.','so she can get a better idea of what the condition entails?']
#sentences = ['why do you feel this way?', 'Let me know if you want to talk.']
candidate = 'Have you thought of directing her to sites like NAMI and Mental Health First Aid'
candidate2 = ' I have been on and off medication for the majority of my life '

cadidates = [candidate, candidate2]

print(f'here we use device: {device}')
coherence_classifier = CoherenceClassifier(device, model_path=politeness_clf_fp)

# #### ####
# (logits, predictions,) = coherence_classifier.predict_empathy(original_responses, candidate)

# print(logits, predictions)

# (logits, predictions,) = coherence_classifier.predict_empathy(original_responses, candidate2)
# print(logits, predictions)


politeness_classifier = PolitenessClassifier(device, model_path=politeness_clf_fp) # sanity check
refutation_classifier = RefutationClassifier(device, model_path=refutation_clf_fp) # sanity check
evidence_classifier = EvidenceClassifier(device, model_path=evidence_clf_fp) # sanity check

(logits, predictions,) = politeness_classifier.predict_empathy(original_responses, candidate2)
print(logits, predictions)


# ==== for refutation classifier ====
print('start sanity check for refutation')
(logits, predictions,) = refutation_classifier.predict_empathy(original_responses, cadidates)
print('the sanity check of refutation classifier version 2')
print(logits, predictions)


print('start sanity check for refutation')
(logits, predictions,) = evidence_classifier.predict_empathy(original_responses, cadidates)
print('the sanity check of evidence classifier version 2')
print(logits, predictions)