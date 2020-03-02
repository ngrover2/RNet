from spacy.lang.en import English
tokenizer = English().tokenizer
from config import base_config
import sys
import torch
from torch import nn
from datetime import datetime
import random as rand
import os
import traceback
import pickle
import ujson
import numpy as np

class RNetConfig():
    def __init__(self):
        self.files_config = dict() # Set directory/file path names
        self.model_config = dict()
        glove_word_embed_dir,word_embed_file_name,word_embed_file_path = None,None,None
        
        # set word_embed_file_path using params in base_config
        if base_config.get("word_embed_file_path", None):
            word_embed_file_path = base_config.get("word_embed_file_path")
        else:
            if base_config.get("glove_embed_dir",None):
                glove_word_embed_dir = base_config["glove_word_embed_dir"]
            if base_config.get("word_embed_file_name",None):
                word_embed_file_name = base_config["word_embed_file_name"]
            word_embed_file_path = os.path.join(glove_word_embed_dir, word_embed_file_name)
        self.files_config['word_embed_file_path'] = word_embed_file_path
        # set logs directory path
        self.files_config['log_dir'] = base_config.get('log_dir','logs')
        # set file path for logs about creating word2index and index2word dictionaries
        vocab_dicts_log_file_name = base_config.get('vocab_dicts_log_file_name','id2word_word2id_dict_creation.log')
        self.files_config['vocab_dicts_log_file'] = os.path.join(self.files_config['log_dir'], vocab_dicts_log_file_name)
        load_training_set_log_file_name = base_config.get('load_training_set_log_file_name','load_training_set.log')
        load_dev_set_log_file_name  = base_config.get('load_dev_set_log_file_name','load_dev_set.log')
        self.files_config['load_training_set_log_file'] = os.path.join(self.files_config['log_dir'], load_training_set_log_file_name)
        self.files_config['load_dev_set_log_file'] = os.path.join(self.files_config['log_dir'], load_dev_set_log_file_name)
        save_dict_dir = base_config.get('save_dict_dir', 'embedding_dicts')
        self.files_config['save_dict_dir'] = save_dict_dir
        id2word_dict_file_name = base_config.get('id2word_dict_file_name', 'id2word_dict')
        word2id_dict_file_name = base_config.get('word2id_dict_file_name', 'word2id_dict')
        self.files_config['id2word_dict_file'] = os.path.join(save_dict_dir, id2word_dict_file_name + '.pkl')
        self.files_config['word2id_dict_file'] = os.path.join(save_dict_dir, word2id_dict_file_name + '.pkl')
        
        dataset_dir = base_config.get('dataset_dir', os.path.abspath(os.path.join(os.path.abspath(''), 'dataset_squad')))
        
        self.files_config['dev_dataset_file'] = base_config.get("dev_dataset_file", None)
        self.files_config['training_dataset_file'] = base_config.get("training_dataset_file")

        try:
            dirs_list = [ 
                self.files_config['save_dict_dir'], 
                self.files_config['log_dir'] + '/' + 'training',
                base_config.get("latest_training_checkpoint_dir", './latest_checkpoint/training'), 
                base_config.get("latest_dev_checkpoint_dir", './latest_checkpoint/dev'),
                base_config.get("saved_object_states_dir", "./saved_objects"),
                os.path.abspath(os.path.join(base_config.get("saved_object_states_dir", './saved_objects'), "training")),
                os.path.abspath(os.path.join(base_config.get("saved_object_states_dir", './saved_objects'), "dev")),
                base_config.get("human_readable_training_dataset_dir", './dataset_readable/training'),
                base_config.get("human_readable_dev_dataset_dir", './dataset_readable/dev'),
                base_config.get("trained_model_dir", None),
                base_config.get("pred_dir", './predictions')
            ]
            self.create_directories_if_not_exist(dirs_list)
        except:
            print(f"Error creating required directories\n{traceback.format_exc()}")
            sys.exit()
        
        self.model_config['word_embedding_dim'] = base_config.get("word_embedding_dim", 300)
        self.model_config['training_batch_size'] = base_config.get("training_batch_size", 32)
        
    
    def create_directories_if_not_exist(self, dirs_list):
        for the_dir in dirs_list:
            if the_dir is not None:
                if not os.path.exists(the_dir) and not os.path.isdir(the_dir):
                    os.makedirs(the_dir)
                    print(f"Created {the_dir}")
    

class SquadDataLoader():
    class SquadDataset():
        def __init__(self):
            self.passages = []
            self.questions = []
            self.answers = []
            self.examples = []
            # self.batches = []

    def __init__(self, rnet_config=None):
        self.id2word_dict = dict()
        self.word2id_dict = dict()
        self.config = RNetConfig()
        self.word_embeddings = None
        self.training_set = self.SquadDataset()
        self.dev_set = self.SquadDataset()
    
    def load_embeddings_and_create_vocab_dictionaries(self):
        id2word_dict = self.id2word_dict
        word2id_dict = self.word2id_dict
        id2word_dict_file = self.config.files_config['id2word_dict_file']
        word2id_dict_file = self.config.files_config['word2id_dict_file']
        curr_id = 0  # curr_id => index of words in vocabulary
        
        # counters for sanity checking stats
        line_counter = 0
        error_count = 0
        too_many_errors = False
        log_dir = self.config.files_config['log_dir']
        log_file = os.path.join(log_dir, 'id2word_word2id_dict_creation.log')
        
        list_word_embeddings = []
        ptrn_word_embedding_size = 300

        # OOV : Out of Vocabulary
        OOV_words = ['PAD','BLANKPH','UNK'] # BLANKPH -placeholder for when no answer is possible
        for word in OOV_words:
            id2word_dict[curr_id] = word
            word2id_dict[word] = curr_id
            list_word_embeddings.append(list(float(0) for _ in range(ptrn_word_embedding_size)))
            curr_id += 1
    
        # Create id2word_dict, word2id_dict and load word-embeddings in nn.Embedding wrapper for easy lookup later
        with open(self.config.files_config['word_embed_file_path'], 'r') as w_emb_file, open(self.config.files_config['vocab_dicts_log_file'], 'w+') as logf:
            for line in w_emb_file:
                line_counter += 1
                try:
                    splits = line.split(" ")
                    word = splits[0]
                    word = word.lower()
                    list_word_embeddings.append(list(float(x) for x in splits[1:]))
                    id2word_dict[curr_id] = word
                    word2id_dict[word] = curr_id
                    curr_id += 1
                except:
                    error_count += 1
                    logf.write(f"\nError generating index for embedding on line: {line}. ERROR TEXT: {traceback.format_exc()}")
                if error_count > 100:
                    too_many_errors = True
                    break
                if line_counter >= 100:
                    break

            if too_many_errors:
                print(f"\nToo many errors! Something is not right. Please inspect the log file - {log_file}")
            else:
                logf.write(f"\nTotal {line_counter} word-embeddings in file")
                logf.write(f"\nSuccesfully read {curr_id - len(OOV_words)} word-embeddings")
                logf.write(f"\nSaving  id2word and word2id dictionaries")
                self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(np.asarray(list_word_embeddings, dtype=np.float32)))
                logf.write(f"\nLoaded nn.Embedding from word-embeddings in file")
                del list_word_embeddings
                with open(id2word_dict_file, 'wb+') as i2w_bfile, open(word2id_dict_file, 'wb+') as w2i_bfile:
                    pickle.dump(id2word_dict, i2w_bfile)
                    logf.write(f"\nid2word dictionary saved at {os.path.abspath(id2word_dict_file)}")
                    pickle.dump(word2id_dict, w2i_bfile)
                    logf.write(f"\nword2id dictionary saved at {os.path.abspath(word2id_dict_file)}")
                    print("Created and saved id2word and word2id dictionaries")
    
    def load_word2id_andid2word(self):
        print(f"Trying to load the id2word_dict_file and word2id_dict_file files")
        id2word_dict_file = self.config.files_config['id2word_dict_file']
        word2id_dict_file = self.config.files_config['word2id_dict_file']
        if self.word_embeddings is not None and os.path.exists(id2word_dict_file) and os.path.exists(word2id_dict_file):
            with open(id2word_dict_file,'rb') as i2w_f, open(word2id_dict_file, 'rb') as w2i_f:
                self.word2id_dict = pickle.load(w2i_f)
                self.id2word_dict = pickle.load(i2w_f)
        else:
            print(f"id2word_dict_file/word2id_dict_file do not exist, creating them afresh.")
            self.load_embeddings_and_create_vocab_dictionaries()

    def load_dataset(self, dataset_type='training', n_examples=None, file_path=None):
        pqa_list = []
        pqa_file = os.path.abspath(os.path.join(base_config.get("saved_object_states_dir", './saved_objects'), f'{dataset_type}/list_PQAs.pkl'))
        max_passage_len = base_config.get("max_passage_len" ,300)
        dataset_file_abs_path = self.config.files_config['training_dataset_file']
        log_file = self.config.files_config['load_training_set_log_file']
        dataset = self.training_set
        if dataset_type == "dev":
            dataset_file_abs_path = self.config.files_config['dev_dataset_file']
            log_file = self.config.files_config['load_dev_set_log_file']
            dataset = self.dev_set

        if file_path is not None:
            print(f"Will load dataset from {file_path}")
            dataset_file_abs_path = os.path.abspath(file_path)

        word2id_dict = self.word2id_dict
        id2word_dict = self.id2word_dict
        data = []
        error_count = 0
        print(f"Loading {dataset_type} data...")
        with open(dataset_file_abs_path, 'r') as ds_file:
            for line in ds_file:
                all_data  = ujson.loads(line)
                data = all_data['data']

        # Load dataset
        include_impossible_questions = True # some questions in the dataset have no answer(s), this flag includes/excludes them from the dataset
        # sequence numbers for question, passage(referred to as 'context' in SQuAD dataset), and answer
        curr_q_num, curr_p_num, curr_a_num = 0,0,0
        
        # flag for premature exit out of the loops (avoid loading whole dataset) for testing or for ad-hoc code analysis
        exit_on_n_examples = -1
        exit_completely = False
        if n_examples is not None and isinstance(n_examples, int) and n_examples > 0:
            exit_on_n_examples = n_examples # number of examples to load before exiting

        human_readable_dataset_dir = base_config.get(f"human_readable_{dataset_type}_dataset_dir", f"./dataset_readable/{dataset_type}")
        self.config.create_directories_if_not_exist([human_readable_dataset_dir])
        human_readable_dataset_file = os.path.join(human_readable_dataset_dir, f"{dataset_type}_dataset.txt")
        with open(log_file, 'w+') as logf, open(log_file, 'w+') as logf, open(human_readable_dataset_file,'w+') as tdata_file:
            for sample in data:
                if exit_completely:
                    break
                for paragraph in sample['paragraphs']:
                    if exit_completely:
                        break
                    passage_text = paragraph['context']
                    passage_text = 'BLANKPH ' + passage_text
                    passage_truncated = False
                    tdata_file.write("[Passage]\n")
                    chars_in_line = 150
                    num_lines = int(len(passage_text) / chars_in_line) + 1
                    for n in range(num_lines):
                        start_pos = n*chars_in_line
                        tdata_file.write(f"{passage_text[ start_pos : start_pos + chars_in_line]}\n")
                    tdata_file.write("\n")
                    try:
                        # paragraph is a dict object
                        tokenised_passage_text = tokenizer(passage_text)
                        passage_tokens = [token.text.lower() for idx,token in enumerate(tokenised_passage_text) if idx <= max_passage_len + 1 ]
                        if len(passage_tokens) < len(tokenised_passage_text):
                            passage_truncated = True
                        # dataset.passages.append(list(word2id_dict.get(token, word2id_dict['UNK']) for token in passage_tokens)) # text
                        # Strategy to find indices of original answer text from tokenized passage:
                        # Insert a marker within the original text to indicate answer start position
                        # so that after tokenising, while searching for a token, if there has not been the marker encountered yet, token should be discarded
                        # Using marker token - MRKTK and inserting it at the answer's start pos
                        
                        # passage_tokens_with_spans = self.token_start_stop_pos(passage_tokens, passage_text)
                        # key['qas'] is a list of question objects
                        for qa in paragraph['qas']: # each question object is a dict
                            answer_beyond_truncation = False
                            if exit_completely:
                                break
                            q_id = qa['id']
                            if qa['is_impossible']:
                                if include_impossible_questions:
                                    dataset.answers.append([0, 0])
                                    dataset.examples.append((curr_p_num, curr_q_num, curr_a_num))
                                    # Save the processed (q_id, passsage text, question text, original answer text, reconstructed answer text, is_passage_truncated, answer_matches_original_exactly, answer_lies_beyomd_truncation_in_passage, reconstructed_answer_found_using_coersion(due to the effects of tokeniser having generated different tokens for answer and passage) tuple for human understandable evaluation later
                                    pqa_list.append( (q_id, passage_text,  qa['question'], "", "", passage_truncated, True, False))
                                    curr_a_num += 1
                                else:
                                    continue
                            else:
                                for answer in qa['answers']: # list of answers
                                    answer_beyond_truncation = False
                                    if exit_completely:
                                        break
                                    ans_text = answer['text']
                                    ans_start_pos = answer['answer_start'] + len('BLANKPH ')
                                    res = self.get_answer_indices(ans_text,ans_start_pos,passage_text,passage_tokens)
                                    if len(res) == 5:
                                        actual_answer_indices = res[0] # tuple of start_index, end_index of tokenised passage list
                                        # make answer span (0,0) if answer lies beyond max_passage_len. If part of answer lies in truncated passage, make answer span (start_answer_index, max_passage_index)
                                        if actual_answer_indices and len(actual_answer_indices) > 0:
                                            if actual_answer_indices[0] > max_passage_len:
                                                answer_beyond_truncation = True
                                                actual_answer_indices = [0, 0]
                                            elif actual_answer_indices[0] <= max_passage_len:
                                                actual_answer_indices = [actual_answer_indices[0], min([actual_answer_indices[-1], max_passage_len])]
                                        else:
                                            actual_answer_indices = [0, 0]

                                        dataset.answers.append([actual_answer_indices[0], actual_answer_indices[-1]])
                                        dataset.examples.append((curr_p_num, curr_q_num, curr_a_num))
                                        curr_a_num += 1
                                        tdata_file.write("<<<<<<Questions>>>>>>>\n")
                                        tdata_file.write(f"{qa['question']}\n")
                                        tdata_file.write(f"Original answer: {res[1]}\n")
                                        tdata_file.write(f"Constructed answer: {res[2]} , with indices {res[0]} in the tokenised text\n")
                                        if res[3] == False: # can also be none, thus the '== False'
                                            tdata_file.write(f"NOTE: Answers may not match exactly\n")
                                        tdata_file.write("--------------------------------------------------------------------------------------------------------------------------------------------\n")
                                        # Save the processed (q_id, passsage text, question text, original answer text, reconstructed answer text, is_passage_truncated, answer_matches_original_exactly, answer_lies_beyomd_truncation_in_passage, reconstructed_answer_found_using_coersion(due to the effects of tokeniser having generated different tokens for answer and passage) tuple for human understandable evaluation later
                                        pqa_list.append( (q_id, passage_text,  qa['question'], res[1], res[2], passage_truncated, res[3], answer_beyond_truncation))
                                        if exit_on_n_examples > 0 and curr_a_num >= exit_on_n_examples:
                                            exit_completely = True
                            if error_count >= 5:
                                exit_completely = True
                                print(f"Errors occurred. Check log file {log_file}")
                            dataset.questions.append(list(word2id_dict.get(token.text.lower(), word2id_dict['UNK']) for token in tokenizer(qa['question']))) # text
                            curr_q_num += 1
                        dataset.passages.append(list(word2id_dict.get(token, word2id_dict['UNK']) for token in passage_tokens)) # text
                        curr_p_num += 1
                    except:
                        logf.write(f"\nError in paragraph:\n----------\n{paragraph}\n----------\n{traceback.format_exc()}")
                        error_count += 1
                        if error_count > 5:
                            print(f"More than 5 errors occurred, stopping. Check log file {log_file}")
                            exit_completely = True
            now = datetime.now().strftime('at %HH %MM %s s on %a, %d-%b-%Y')
            logf.write(f"{dataset_type} set generated {now} containing {len(dataset.examples)} examples")
            
            try:
                with open(pqa_file, 'wb+') as pf:
                    pickle.dump(pqa_list, pf)
            except:
                raise Exception(f"pqa_file could not be saved at {pqa_file}.\n{traceback.format_exc()}")

    def get_answer_indices(self, ans_text, ans_start_pos, passage_text, passage_tokens):
        ans_end_pos = ans_start_pos + len(ans_text)
        ans_text_tokens = list(token.text for token in tokenizer(ans_text))

        # var to hold constructed answer text after tokenisation, for debugging whether the preprocessing was successful
        final_ans_text = ""
        answers_maybe_diff = None
        
        # find answer tokens' spans
        # check if all anwer tokens are there at consecutive positions in list
        # may not be true as the anwer tokens present in the list may have some characters inserted between them
        # May be insert a marker within the original text to indicate answer token start position 
        # so that when tokenising, if there has not been the marker encountered yet, token should be discarded
        
        # Insert marker
        passage_text_mod = passage_text[:ans_start_pos] + 'MRKTK ' + passage_text[ans_start_pos:]
        # Tokenize passage text with marker
        passage_tokens_mod = [token.text for token in tokenizer(passage_text_mod)]
        # Find answer spans from passage text with marker
        passage_tokens_mod_with_spans = self.token_start_stop_pos(passage_tokens_mod, passage_text_mod)
        
        # find index and span of marker token
        idx_MRKTK = -1
        span_MRKTK = None
        for idx, (token, pos_span) in enumerate(passage_tokens_mod_with_spans):
            if token == 'MRKTK':
                idx_MRKTK = idx
                span_MRKTK = pos_span

        # Now find answer spans and answer tokens in 
        # passage text with marker token included (passage_text_mod) using :
        # 1. passage_tokens_mod and
        # 2. passage_text_mod's tokens with its char-positions spans - passage_tokens_mod_with_spans
        ans_tk_offset = 0 # offset of current (in the loop below) token within answer
        ans_idxs = [] # to hold the found answer indices
        # print(passage_text_mod)
        # print(idx_MRKTK)
        answer_found_using_coersion = False
        find_within = 3 # find within 3 tokens from starting token
        for ans_tk in ans_text_tokens:
            ans_tk = str(ans_tk)
            err = ""
            # find answer token in - passage text with marker token
            try:
                found_at_idx = passage_tokens_mod.index(ans_tk, idx_MRKTK + ans_tk_offset)
                if found_at_idx <= idx_MRKTK + ans_tk_offset + find_within:
                    ans_idxs.append(found_at_idx)
            except ValueError:
                # Tokenisation process would not necessarily produce the same tokens as they are in actual text. 
                # Thefefore, we need to first search the tokenised version of the answer token by looking at the tokenised paragraph tokens and 
                # extracting the first occurence of the answer token that is fully contained in the tokenised paragraph token. 
                # Then find the index of the tokenised paragraph token
                tokenised_p_tokens = passage_tokens_mod[idx_MRKTK + ans_tk_offset:]
                trial_cnt = 0
                for t_pos, t_p_tk in enumerate(tokenised_p_tokens):
                    trial_cnt += 1
                    if str(ans_tk) in t_p_tk:
                        ans_tk = t_p_tk
                        try:
                            answer_found_using_coersion = True
                            found_at_idx = passage_tokens_mod.index(ans_tk, idx_MRKTK + ans_tk_offset)
                            if found_at_idx <= idx_MRKTK + ans_tk_offset + find_within:
                                ans_idxs.append(found_at_idx)
                            else:
                                break
                        except ValueError:
                            err += f"\n{traceback.format_exc()}"
                            err += f"\nIndexof: {ans_tk} not found in {passage_text_mod[ans_start_pos:]}\nor in \n{tokenised_p_tokens}\nAnswer was: {ans_text_tokens}\n"
                            raise Exception(err)
                        except:
                            raise
                        finally:
                            break
            except:
                raise
            ans_tk_offset += 1
        
        actual_answer_indices = (0,0) # default, if indices were not found or there is no answer
        if len(ans_idxs) > 0:
            # answer indices found above (ans_idxs) are from passage text that includes the marker token - MRKTK
            # Therefore, Offset the answer indices by -1 to account for it and get indices in the original passage text
            actual_answer_indices = (ans_idxs[0] - 1, ans_idxs[-1] - 1)
            
            # Constructed answer
            final_ans_text = " ".join(passage_tokens_mod[ans_idxs[0]:ans_idxs[-1] + 1 ])
            if answer_found_using_coersion:
                # If our constructed answer is off by a lot, (comparing the size of answers),
                # then we ignore the answer and say that it is not answerable.
                # This can be avoided with more involved pre-processing.
                if (len(final_ans_text) > 25 and  (len(final_ans_text) / max(len(ans_text), 1) ) > 1.5 ):
                    return ((0,0), ans_text, final_ans_text, True)
                else:
                    answers_maybe_diff = False
        else:
            pass
        
        return (actual_answer_indices, ans_text, final_ans_text, answers_maybe_diff, answer_found_using_coersion)

    def load_inference_batch(self, passage='', questions=[]):
        
        questions_tokenised = []
        passage_tokenised = []
        word2id_dict = self.word2id_dict

        if passage == '' or passage is None:
            print("No Passage(context) submitted")
            return
        if not questions or len(questions) ==  0:
            print("No questions submitted")
            return

        batch_size = len(questions)
        # passage_tokenised ,questions_tokenised holds list of the list of word_ids for each passage/questions
        # so length of passage_tokenised/questions_tokenised is the number of passages/questions resp.
        passage_tokens = [token.text for token in tokenizer(passage)]
        passage_tokenised.append(list(word2id_dict.get(token.lower(), word2id_dict['UNK']) for token in passage_tokens)) # text
        for question in questions:
            questions_tokenised.append(list(word2id_dict.get(token.text.lower(), word2id_dict['UNK']) for token in tokenizer(question) )) # text

        max_ques_len , max_passage_len = 0,0
        word_embedding_dim = self.config.model_config['word_embedding_dim'] = 300
        passage_wemb, ques_wemb = [],[]
        # passage_wemb,ques_wemb holds list of word-embeddings for all passages/questions
        passage_wemb.append(self.word_embeddings(torch.tensor(passage_tokenised[0]).long()))
        
        max_passage_len = max(len(passage_tokenised[0]), max_passage_len) # passage_tokenised[0] holds the word_ids for first passage's words
        
        # there should be only one passage and possibly multiple questions given by len(questions_tokenised)
        for idx in range(len(questions_tokenised)):
            q_ids = questions_tokenised[idx]
            ques_wemb.append(self.word_embeddings(torch.tensor(q_ids).long()))    # append tensor of shape(lQ,word_embedding_dim)
            max_ques_len = max(len(q_ids), max_ques_len)
        
        passage_words_padded = torch.zeros((max_passage_len, batch_size, word_embedding_dim)) # placeholder tensor # (max_passage_len, 1, word_embedding_dim)
        ques_words_padded = torch.zeros((max_ques_len, batch_size, word_embedding_dim)) # placeholder tensor # (max_ques_len, 1, word_embedding_dim)
        
        for i in range(batch_size):
            curr_passage_len = passage_wemb[0].shape[0] # lP
            curr_ques_len = ques_wemb[i].shape[0] # lQ
            passage_words_padded[:curr_passage_len,i, :] = passage_words_padded[i] # (max_passage_len, 1, word_embedding_dim)
            ques_words_padded[:curr_ques_len, i, :] = ques_words_padded[i] # (max_ques_len, 1, word_embedding_dim)
        
        return (passage_words_padded, ques_words_padded)


    def token_start_stop_pos_(self, tokens, passage):
        words_start_stop_pos = []
        from_pos = 0
        for token in tokens:
            start_pos = passage.find(token, from_pos)
            if start_pos >= 0:
                words_start_stop_pos.append( (start_pos, start_pos + len(token)) )
        return words_start_stop_pos
    
    def token_start_stop_pos(self, tokens, passage):
        words_start_stop_pos = []
        from_pos = 0
        for token in tokens:
            start_pos = passage.find(token, from_pos)
            if start_pos >= 0:
                words_start_stop_pos.append( (token, (start_pos, start_pos + len(token)) ) )
        return words_start_stop_pos
    
    def map_indices_to_data_in_batch(self, batch, batch_dataset_type='training'):
        # we need to know the max seq size in curr batch, so that we can calculate the size of the padded tensor
        if batch_dataset_type == 'dev':
            dataset = self.dev_set
        else:
            dataset = self.training_set

        max_ques_len ,max_passage_len = 0,0
        batch_size = len(batch)
        word_embedding_dim = self.config.model_config['word_embedding_dim'] = 300
        answer_spans = torch.zeros((batch_size, 2))
        b_wrd_passages, b_wrd_questions = [],[]
        # Save training/dev example's index (example/datapoint's number) from original data, to verify predicted answers while training/testing/debugging
        ex_idx = []
        for idx, (passage_idx, question_idx, answers_idx) in enumerate(batch):
            word_ids_in_passage = dataset.passages[passage_idx]
            b_wrd_passages.append(self.word_embeddings(torch.tensor(word_ids_in_passage).long()))

            word_ids_in_question = dataset.questions[question_idx]
            b_wrd_questions.append(self.word_embeddings(torch.tensor(word_ids_in_question).long()))
            
            answer_spans[idx,:] = torch.FloatTensor(dataset.answers[answers_idx])
            
            max_passage_len = max(len(word_ids_in_passage), max_passage_len)
            max_ques_len = max(len(word_ids_in_question), max_ques_len)        
            ex_idx.append(answers_idx)
        
        passage_words_padded = torch.zeros((max_passage_len, batch_size, word_embedding_dim)) # placeholder tensor
        ques_words_padded = torch.zeros((max_ques_len, batch_size, word_embedding_dim)) # placeholder tensor
        
        for i in range(batch_size):
            curr_passage_len = b_wrd_passages[i].shape[0]
            curr_ques_len = b_wrd_questions[i].shape[0]
            passage_words_padded[:curr_passage_len,i, :] = b_wrd_passages[i]
            ques_words_padded[:curr_ques_len, i, :] = b_wrd_questions[i]
        
        return (passage_words_padded, ques_words_padded, answer_spans, ex_idx)
    
    def construct_batches(self, dataset_type='training', batch_size=None):
        if batch_size is None:
            raise Exception("construct_batches got a batch size of None.")
        dataset = self.training_set
        if dataset_type == 'dev':
            dataset = self.dev_set
        
        batches = []
        num_examples = len(dataset.examples)

        if num_examples < 1:
            print("No examples in dataset.")
            return

        print(f"Shuffling {num_examples} examples in dataset")
        rand.shuffle(dataset.examples)
        
        for b in range(0, num_examples, batch_size):
            this_batch = dataset.examples[b: b + batch_size]
            batches.append(this_batch)
        last_batch_size = len(batches[-1])
        last_batch_short = batch_size - len(batches[-1])
        if batch_size - last_batch_size > 0:
            batches[-1].extend(list((dataset.examples[np.random.randint(0, num_examples)] for i in range(last_batch_short))))
        
        print(f"Shuffling {len(batches)} batches in dataset")
        rand.shuffle(batches)
        
        return batches
    
    def get_one_batch(self, dataset_type='training'):
        if dataset_type == 'dev':
            dataset = self.dev_set
        else:
            dataset = self.training_set
        # get a random batch from the dataset
        if not dataset.batches or len(dataset.batches) < 0:
            self.construct_batches(dataset_type)
        rand.shuffle(list(range(len(dataset.batches))))
        return dataset.batches[0]