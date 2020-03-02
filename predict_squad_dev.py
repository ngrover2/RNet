from argparse import ArgumentParser
import os
import sys
import time
import traceback
import ujson
from config import base_config
import torch
import torch.nn.functional as F
from data_utils import SquadDataLoader
from rnet_model import RNetModel
import pickle
from spacy.lang.en import English
tokenizer = English().tokenizer

dataset_dir = base_config.get("dataset_dir", None)

trained_model_dir = base_config.get("trained_model_dir", None)
if trained_model_dir is None:
	print("Directory name where trained RNET model(s) are stored (trained_model_dir) is not defined in config.")
	raise SystemExit


predictions_dir = base_config.get("pred_dir", None)
if predictions_dir is None:
	print("Directory name where trained predictions will be stored (predictions_dir) is not defined in config.")
	raise SystemExit


def path_includes_directory(path=None):
	if path is None:
		return
	try:
		return not os.path.split(path)[0] == ""
	except:
		return

def create_dir_if_not_exists(path=None):
	if path is None:
		raise Exception("Can not create empty directory. create_dir_if_not_exists received an empty path")
	path_splits = os.path.split(path)
	pdir = path_splits[0]
	pfile = path_splits[1]
	# create directory if not exists
	if not(os.path.exists(pdir) and os.path.isdir(pdir)):
		try:
			os.makedirs(pdir)
		except:
			print(f"Error occurred while creating directory {pdir}. Please provide an existing directory path.\n{traceback.format_exc()}")
			raise SystemExit

def validate_args(arguments=None):
	print(f"Arguments help:\n \
1. Use -o or --outputfile for output file name/filepath where predictions will be stored\n\
2. Use -m or --modelfile for the file name of the pytorch (.pt) model to load for prediction\n\
3. (Optional, if dataset_dir and dataset_dev_file_name are already defined in config) Use -d or --datafile for the file name/path of the Squad dev dataset json file\n")

	o_abs_path = None
	m_abs_path = None
	d_abs_path = None

	if arguments.outputfile is None:
		print(f"Please provide output file name where predictions will be stored (using the -o or --output flag)")
		raise SystemExit
	if arguments.modelfile is None:
		print(f"Please provide the file path of the pytorch (.pt) model to load for prediction (using -m or --modelfilepath) flag. Note that the program will look for the file in the directory as defined by the `trained_model_dir` key in the config file")
		raise SystemExit

	if isinstance(arguments.outputfile, str):
		try:
			if path_includes_directory(arguments.outputfile):
				o_abs_path = os.path.abspath(arguments.outputfile)
			else:
				o_abs_path = os.path.abspath(os.path.join(predictions_dir, arguments.outputfile))
			create_dir_if_not_exists(o_abs_path)
		except:
			print(f"Outputfile path - {arguments.outputfile} is invalid")
			raise SystemExit
	
	if isinstance(arguments.modelfile, str):
		print(f"Received arguments.modelfile:{arguments.modelfile}")
		try:
			if path_includes_directory(arguments.modelfile):
				print(f"Path includes a directory")
				m_abs_path = os.path.abspath(arguments.modelfile)
			else:
				print(f"Path does not include a directory")
				m_abs_path = os.path.abspath(os.path.join(trained_model_dir, arguments.modelfile))
		except:
			print(f"RNET Model file - {arguments.modelfile} is invalid")
			raise SystemExit
	
	if arguments.datafile and isinstance(arguments.datafile, str):
		try:
			if path_includes_directory(arguments.datafile):
				d_abs_path = os.path.abspath(arguments.datafile)
				if not(os.path.exists(d_abs_path) and os.path.isfile(d_abs_path)):
					print(f"No file found for the Squad Dev Dataset at path {d_abs_path}")
					raise SystemExit
			else:
				if dataset_dir is None:
					print("Squad Dataset directory (dataset_dir) is not defined in config. Please either provide the dir path in config or provide absolute path of the dev data json file using the -d or --datafile flag")
					raise SystemExit
				else:
					d_abs_path = os.path.abspath(os.path.join(dataset_dir, arguments.datafile))
		except:
			print(f"RNET Model file - {arguments.modelfile} is invalid")
			raise SystemExit
	
	return (d_abs_path, o_abs_path, m_abs_path)
	

aparser = ArgumentParser() 
aparser.add_argument('-o', '--outputfile') # Please provide output file name/path where predictions will be stored
aparser.add_argument('-m', '--modelfile') # Please provide file name/path of the pytorch (.pt) model to load for prediction
aparser.add_argument('-d', '--datafile') # Please provide file name/path of the Squad dev set json file

arguments = aparser.parse_args()
d_abs_path, o_abs_path, m_abs_path = validate_args(arguments)

if d_abs_path is None:
	dataset_dev_file_name = base_config.get("dataset_dev_file_name", None)
	d_abs_path = os.path.abspath(os.path.join(dataset_dir, dataset_dev_file_name))
	if not(os.path.exists(d_abs_path) and os.path.isfile(d_abs_path)):
		print(f"No file found for the Squad Dev Dataset at path {d_abs_path}")
		raise SystemExit

dataset_type = "dev"
dev_n_examples = base_config.get("dev_n_examples", None)
dev_batch_size = base_config.get("dev_batch_size", 32)
use_gpu = base_config.get("use_gpu", False)
pqa_dev_file = os.path.abspath(os.path.join(base_config.get("saved_object_states_dir", './saved_objects'), f'dev/list_PQAs.pkl'))
pqa_dev_list = []

m_file_name = os.path.splitext(os.path.split(m_abs_path)[1])[0]
pred_log_dir = os.path.join(base_config['log_dir'] , f"{dataset_type}")
pred_log_file = os.path.join(pred_log_dir, f'pred_dev_{m_file_name}.log')
arguments = aparser.parse_args()
print(f"Squad dev dataset will be loaded from {d_abs_path}\n")
print(f"Saved Model weights will be loaded from {m_abs_path}\n")
print(f"Output will be saved to {o_abs_path}\n")

input("If this seems ok, Press any key to continue. Otherwise press ctrl+C to abort")

try:
	predictions = dict()
	device = torch.device("cuda" if torch.cuda.is_available() and use_gpu == True else "cpu")
	rnet = RNetModel(batch_size=dev_batch_size)
	
	with open(pred_log_file, 'w+') as log_file:
		print(f"Loading {dev_n_examples or 'All'} examples from dev dataset..")
		print(f"Loading {dev_n_examples or 'All'} examples from dev dataset..", file=log_file)
        
		dataloader = SquadDataLoader()
		dataloader.load_word2id_andid2word()
		dataloader.load_dataset(f"{dataset_type}", dev_n_examples, d_abs_path)

		if dataloader is None or len(dataloader.dev_set.examples) < 1:
			print(f"Error loading data through dataloader (dataloader.dev_set.examples) \n{traceback.format_exc()}", file=log_file)
			print(f"Error loading data through dataloader. Please look at the log file {pred_log_file}")
			raise SystemExit
		
		try:
			with open(pqa_dev_file, 'rb') as pqafile:
				pqa_dev_list = pickle.load(pqafile)
				print(f"Loaded passage questions answers (pqa)_list which has {len(pqa_dev_list)} examples")
		except:
			print(f"Passage questions answers (pqa)_list can not be loaded \n{traceback.format_exc()}", file=log_file)
		
		print("Creating Model..", file=log_file)
		print("Creating RNETModel..")
		
		if use_gpu:
			rnet.cuda()
			print("Will use GPU")
		
		try:
			print("Loading trained weights from saved model..", file=log_file)
			print("Loading trained weights from saved model..")
			rnet.load_state_dict(torch.load(m_abs_path))
			rnet.eval()
		except:
			print(f"Rnet pytorch model can not be loaded from {m_abs_path}\n{traceback.format_exc()}")
			raise SystemExit
		
		exit_after_n_errs = 2
		errs = 0
		loss = 0
		with torch.no_grad():
			print(f"Constructing batches for {dataset_type} set..")
			print(f"Constructing batches for {dataset_type} set..", file=log_file)
			batches = []
			try:
				batches = dataloader.construct_batches(dataset_type, dev_batch_size)
			except:
				print(f"Could not load batches from {dataset_type} set\n{traceback.format_exc()}")
				print(f"Could not load batches from {dataset_type} set\n{traceback.format_exc()}", file=log_file)
				log_file.flush()
				raise SystemExit
			for b_idx, a_batch in enumerate(batches):
				try:
					passage_words_padded, ques_words_padded, answer_span, ex_idx = dataloader.map_indices_to_data_in_batch(a_batch,f"{dataset_type}")
					
					# output is a tuple of (start_indices_probabilities_tensor, end_indices_probabilities_tensor) each of shape(batch,num_words_in_passage)
					output = rnet(passage_words_padded, ques_words_padded)
					start_indices_pred_prob, pred_start_indices = torch.max(output[0], dim=1)
					end_indices_pred_prob, pred_end_indices = torch.max(output[1], dim=1)
					loss_pos_start = F.cross_entropy(output[0], answer_span[:, 0].long().to(device))
					loss_pos_end = F.cross_entropy(output[1], answer_span[:, 1].long().to(device))

					# predicted answer span vs actual answer span
					for i in range(output[0].shape[0]): # for every element in batch
						# original answer span
						o_start, o_end = answer_span[i,0], answer_span[i,1]
						# predicted answer span
						p_start, p_end = pred_start_indices[i], pred_end_indices[i]
						p_start_prob, p_end_prob = start_indices_pred_prob[i], end_indices_pred_prob[i]
						# ex is a tuple containing (qid, passsage text, question text, original answer text, reconstructed answer text, is_passage_truncated, answer_matches_original_exactly, answer_lies_beyond_truncation_in_passage , answer_found_using_coersion(due to the effects of tokeniser having generated different tokens for answer and passage)
						example_index = None
						ex = None
						try:
							example_index = ex_idx[i]
						except IndexError:
							print(f"ex_idx list does not have a {i} index", file=log_file)
							raise
						try:
							ex = pqa_dev_list[example_index]
						except IndexError:
							print(f"pqa_dev_list does not have a {example_index} index", file=log_file)
							raise
						
						pred_record = dict()
						qid = ex[0]
						orig_ans = ex[4]
						pred_ans = " ".join([token.text for token in tokenizer(ex[1])[p_start:p_end+1]])

						if 'blankph' in pred_ans.lower():
							blk_idx = pred_ans.lower().index('blankph')
							pred_ans = pred_ans.lower()[blk_idx+len('blankph'):]

						predictions[qid] = pred_ans
						print(f"Question:{ex[2]} with qid: [{qid}] \nOriginal: [{orig_ans}] with indices ({o_start},{o_end})\nPredicted: [{pred_ans}] with indices ({p_start},{p_end})\n\n", file=log_file)
						log_file.flush()
					loss += (loss_pos_start + loss_pos_end) / 2
				except:
					errs += 1
					if errs >= exit_after_n_errs:
						raise
					print(f"Error in batch {b_idx}, remaining error tolerance: {exit_after_n_errs-errs}")
			print(f"Dev Loss: {loss}", file=log_file)
			log_file.flush()
	with open(o_abs_path, 'w+') as pfile:
		ujson.dump(predictions, pfile)
		print(f"Predictions saved in {o_abs_path}")
except SystemExit:
	print(f"Error(s) occurred. There may be more details in the log file {pred_log_file}")
	pass
except:
	print(traceback.format_exc())