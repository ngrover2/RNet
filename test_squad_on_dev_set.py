import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import traceback
import sys
import shutil
from shutil import copyfile
import tempfile

from data_utils import SquadDataLoader
from rnet_model import RNetModel
from config import base_config

def get_checkpoint_file_name(dir_name, batch):
    return os.path.join(dir_name, f"test_checkpoint_batch_{batch}.chk")

def save_object_state(file_name=None,obj=None,obj_name=None,log_file=None):
    if file_name is None:
        print("Error:Invalid file name. Please provide a valid file name top save dataloader object state", file=log_file)
        return
    if obj_name is None:
        obj_name = "Unknown"
    torch.save(obj, file_name)
    print(f"{obj_name} state saved at {file_name}", file=log_file)

def create_directories_if_not_exist( dir_path):
    if dir_path is not None:
        if not os.path.exists(dir_path) and not os.path.isdir(dir_path):
            os.makedirs(dir_path)

if __name__ == "__main__":
    try:
        trained_model_dir = base_config.get("trained_model_dir", "./trained_model")
        trained_model_file = os.path.abspath(os.path.join(trained_model_dir, "rnet_trained.pt"))
        if not(os.path.exists and os.path.isfile(trained_model_file)):
            print(f"Trained Model file does not exist at {trained_model_file}")
            raise SystemExit

        use_gpu = base_config.get("use_gpu", False)
        batch_size = base_config.get("dev_batch_size", None)
        n_examples = base_config.get("dev_n_examples", None)

        latest_checkpoint_dir = base_config.get("latest_dev_checkpoint_dir", None)
        saved_objs_dir = base_config.get("saved_object_states_dir", None)
        dataloader_state_file_path = os.path.abspath(os.path.join(saved_objs_dir, f"SquadDataLoader_dev_batchsize{batch_size}_num_examples{n_examples if n_examples else 'All'}.obj"))
        test_afresh = base_config.get("test_afresh", False)
        device = torch.device("cuda" if torch.cuda.is_available() and use_gpu == True else "cpu")
        
        testing_log_dir = os.path.join(base_config['log_dir'] , 'test')
        create_directories_if_not_exist(testing_log_dir)
        testing_log_file = os.path.join(testing_log_dir, 'dev_eval_log_trained_model.txt')
        pqa_file = os.path.abspath(os.path.join(base_config.get("saved_object_states_dir", './saved_objects'), f'dev/list_PQAs.pkl'))
        
        dataloader = None
        num_tries = 2
        try_cnt = 0
        with open(testing_log_file, 'w+') as log_file:
            while(dataloader is None):
                try_cnt += 1
                if try_cnt > num_tries:
                    break
                try:
                    if (test_afresh == True or \
                        (test_afresh == False and \
                            not (
                                os.path.exists(dataloader_state_file_path) and \
                                    os.path.isfile(dataloader_state_file_path)
                                )
                        )): # to avoid other truthy values as it is loaded from config file
                        print("Loading dev dataset..")
                        print("Loading dev dataset..", file=log_file)
                        dataloader = SquadDataLoader()
                        dataloader.load_word2id_andid2word()
                        dataloader.load_dataset('dev', n_examples)
                        print("Constructing batches for dev dataset..")
                        print("Constructing batches for dev dataset..", file=log_file)
                        dataloader.construct_batches('dev', batch_size)
                        save_object_state(dataloader_state_file_path, dataloader,"SquadDataLoader object (for dev set)", log_file)
                    else:
                        if os.path.exists(dataloader_state_file_path) and os.path.isfile(dataloader_state_file_path):
                            print(f"Dataloader state exists\n. Loading state from {dataloader_state_file_path}", file=log_file)
                            dataloader = torch.load(dataloader_state_file_path)
                        else:
                            test_afresh = True
                except:
                    print(traceback.format_exc(), file=log_file)
                    test_afresh = True
                    print("Saved dataloader state (for dev set) may be corrupt as it could not be loaded, will create afresh")
                    continue
            
            if dataloader is None:
                print("Error loading dev set data through dataloader. Please debug", file=log_file)
                raise SystemExit
            
            print("Creating Model..", file=log_file)
            print("Creating Model..")
            rnet = RNetModel(batch_size=batch_size)
            if use_gpu:
                rnet.cuda()
            rnet.eval()
            
            dev_loss = 0
            with torch.no_grad():
                for b_idx, a_batch in enumerate(dataloader.dev_set.batches):
                    if last_batch > 0 and b_idx <= last_batch:
                        print(f"Batch {b_idx} already tested", file=log_file)
                        continue
                    try:
                        passage_words_padded, ques_words_padded, answer_span, ex_idx = dataloader.map_indices_to_data_in_batch(a_batch,'dev')
                        output = rnet(passage_words_padded, ques_words_padded)
                        # output is a tuple of (start_indices_probabilities_tensor, end_indices_probabilities_tensor) each of shape(batch,num_words_in_passage)
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
                            
                            # ex (below) is a tuple containing (passsage text, question text, original answer text, reconstructed answer text, is_passage_truncated, answer_matches_original_exactly, answer_lies_beyomd_truncation_in_passage)
                            example_index = None
                            ex = None
                            try:
                                example_index = ex_idx[i]
                            except IndexError:
                                print(f"ex_idx list does not have a {i} index", file=log_file)
                                raise SystemExit

                            try:
                                ex = pqa_list[example_index]
                                print(f"Question:{ex[1]}\nOriginal VS Predicted Answer Indices => ({o_start}-{o_end}) VS ({p_start},{p_end}) with probs ({p_start_prob},{p_end_prob})", file=log_file)
                                # similar pre-processing as applied while creating answer indices
                            except IndexError:
                                print(f"pqa_list does not have a {example_index} index", file=log_file)
                                raise SystemExit
                        
                        sum_loss = (loss_pos_start + loss_pos_end) / 2
                        dev_loss += sum_loss
                        print(f"Loss - batch{b_idx} {sum_loss}",file=log_file)
                        state = {
                            'batch': b_idx,
                            'state_dict': rnet.state_dict(),
                        }
                        
                        # Checkpoint
                        save_checkpoint(latest_checkpoint_dir, state, epoch, b_idx, test_afresh,log_file)
                        del loss_pos_start, loss_pos_end, output, pred_start_indices, pred_end_indices, start_indices_pred_prob, end_indices_pred_prob, sum_loss
                        
                        # if (b_idx) >= 50:
                        #     torch.cuda.empty_cache()
                        del passage_words_padded, ques_words_padded, answer_span
                    except:
                        print(traceback.format_exc() ,file=log_file)
                        raise SystemExit
    except SystemExit:
        print(f"Please look at the log file at {testing_log_file}")
        pass
    except:
        print(traceback.format_exc())
        print(f"Error occurred, please look at the log file at {testing_log_file}")



    
