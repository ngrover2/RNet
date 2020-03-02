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
import pickle
import ujson
from spacy.lang.en import English
tokenizer = English().tokenizer

from data_utils import SquadDataLoader
# from rnet_model import RNetModel
# from rnet_model2 import RNetModel
# from rnet_model3 import RNetModel
from rnet_model4 import RNet as RNetModel
from config import base_config



def get_checkpoint_file_name(dir_name, epoch):
    return os.path.join(dir_name, f"checkpoint_epoch_{epoch}.chk")

def save_object_state(file_name=None,obj=None,obj_name=None,log_file=None):
    if file_name is None:
        print("Please provide a valid file name top save dataloader object state", file=log_file)
        return
    if obj_name is None:
        obj_name = "Unknown"
    torch.save(obj, file_name)
    print(f"{obj_name} state saved at {file_name}", file=log_file)

def save_checkpoint(chk_dir=None, state=None, epoch=None, save_prev=True, log_file=None):
    tmpfile = None
    if state is None:
        print(f"No state to save for epoch {epoch}", file=log_file)
        return
    with tempfile.TemporaryDirectory() as tmpd:
        for entity in os.listdir(chk_dir):
            if entity[0] == ".":
                del entity
                continue
            if os.path.split(entity)[1].endswith("chk"):
                abs_path = os.path.abspath(os.path.join(chk_dir, entity))
                # copy last chk file to a tmp location
                if save_prev == True:
                    tmpfile = shutil.copy(abs_path,tmpd)
                os.remove(abs_path)
        try:
            checkpoint_file_name = get_checkpoint_file_name(chk_dir, epoch)
            print(f"Saving new checkpoint for the model at {checkpoint_file_name}", file=log_file)
            with open(checkpoint_file_name,'wb+') as sfile:
                torch.save(state, sfile)
        except:
            print("Error saving new checkpoint, restoring prev checkpoint", file=log_file)
            print("Error saving new checkpoint, restoring prev checkpoint")
            if tmpfile is not None:
                shutil.copy(tmpfile, chk_dir)

def load_from_checkpoint(file_name=None, model=None, optimizer=None, log_file=None):
    if file_name is None:
        print("No file name provided for loading model checkpoint")
        print("No file name provided for loading model checkpoint", file=log_file)
        return

    last_epoch = 0
    last_epoch_loss = 0
    if os.path.isfile(file_name):
        print(f"Loading model from checkpoint: {file_name}", file=log_file)
        checkpoint_state = torch.load(file_name)
        last_epoch = checkpoint_state['epoch']
        last_epoch_loss = checkpoint_state.get('last_epoch_loss', 0)
        model.load_state_dict(checkpoint_state['state_dict'])
        optimizer.load_state_dict(checkpoint_state['optimizer'])
        print(f"Loaded from checkpoint saved at epoch {last_epoch}", file=log_file)
    else:
        print(f"Checkpoint file does not exist at {file_name}", file=log_file)

    return model, optimizer, last_epoch, last_epoch_loss

def create_directory_if_not_exists(dir_path):
    try:
        if dir_path is not None:
            if not os.path.exists(dir_path) and not os.path.isdir(dir_path):
                os.makedirs(dir_path)
    except:
        raise Exception(f"Directory {dir_path} can not be created. \n{traceback.format_exc()}\n")

if __name__ == "__main__":
    try:
        dataset_type = 'training'
        use_gpu = base_config.get("use_gpu", False)
        batch_size = base_config.get("training_batch_size", None)
        n_examples = base_config.get("train_n_examples", None)
        if batch_size is None:
            raise Exception(f"{dataset_type} Batch size is not defined in config file (config key: {dataset_type}_batch_size)")
        num_epochs = base_config.get("num_epochs", 10)
        train_afresh = base_config.get("train_afresh", False)
        device = torch.device("cuda" if torch.cuda.is_available() and use_gpu == True else "cpu")
        
        trained_model_dir = base_config.get("trained_model_dir", "./trained_model")
        latest_checkpoint_dir = base_config.get("latest_training_checkpoint_dir", None)
        backup_checkpoint_dir = './checkpoints_bkp'
        saved_objs_dir = base_config.get("saved_object_states_dir", None)
        dataloader_state_file_path = os.path.abspath(os.path.join(saved_objs_dir, f"SquadDataLoader_training_num_examples{n_examples if n_examples else 'All'}.obj"))
        
        training_log_dir = os.path.join(base_config['log_dir'] , 'training')
        training_log_file = os.path.join(training_log_dir, 'training_log.txt')
        training_loss_file = os.path.join(training_log_dir, 'training_loss.txt')
        
        for d in [saved_objs_dir, trained_model_dir, training_log_dir, latest_checkpoint_dir, backup_checkpoint_dir]:
            create_directory_if_not_exists(d)
        
        pqa_file = os.path.abspath(os.path.join(base_config.get("saved_object_states_dir", './saved_objects'), f'training/list_PQAs.pkl'))
        pqa_list = []
                
        dataloader = None
        num_dataloader_tries = 2
        try_cnt = 0
        force_reload = False
        with open(training_log_file, 'w+') as log_file, open(training_loss_file, 'a+') as loss_file:
            while(dataloader is None):
                try_cnt += 1
                if try_cnt > num_dataloader_tries:
                    break
                try:
                    print(f"Loading {dataset_type} dataset..")
                    print(f"Loading {dataset_type} dataset..", file=log_file)
                    if (train_afresh == True or force_reload == True or \
                        (train_afresh == False and \
                            not (
                                os.path.exists(dataloader_state_file_path) and \
                                    os.path.isfile(dataloader_state_file_path)
                                )
                        )): # to avoid other truthy values as it is loaded from config file
                        dataloader = SquadDataLoader()
                        dataloader.load_word2id_andid2word()
                        dataloader.load_dataset(dataset_type, n_examples)
                        # print("Constructing batches..")
                        # print("Constructing batches..", file=log_file)
                        # dataloader.construct_batches('training', batch_size)
                        save_object_state(dataloader_state_file_path, dataloader,f"SquadDataLoader object for {dataset_type} set",log_file)
                    else:
                        try:
                            if os.path.exists(dataloader_state_file_path) and os.path.isfile(dataloader_state_file_path):
                                print(f"Saved Dataloader state found\n. Loading state from {dataloader_state_file_path}", file=log_file)
                                dataloader = torch.load(dataloader_state_file_path)
                            else:
                                print(f"No existing Dataloader state found, will create afresh", file=log_file)
                                force_reload = True
                                continue
                        except:
                            print("Saved dataloader state may be corrupt as it could not be loaded, will create afresh")
                            force_reload = True
                            continue
                except:
                    print(f"Error occurred while loading data through Dataloader\n{traceback.format_exc()}", file=log_file)
                    print(f"Error occurred while loading data through Dataloader\n{traceback.format_exc()}")
                    raise SystemExit
            
            if dataloader is None or len(dataloader.training_set.examples) < 1:
                print(f"Error loading data through dataloader. Please look at the log file {training_log_file}")
                raise SystemExit
            with open(pqa_file, 'rb') as pqafile:
                pqa_list = pickle.load(pqafile)
                # pqa_dev_list = pickle.load(pqatestfile)
                print(f"Loaded pqa_list which has {len(pqa_list)} examples")
                # print(f"Loaded pqa_dev_list which has {len(pqa_dev_list)} examples")

            print("Creating Model..", file=log_file)
            print("Creating Model..")
            rnet = RNetModel()
            if use_gpu:
                rnet.cuda()
            # optimizer = optim.Adadelta(rnet.parameters(), rho=0.95, eps=1e-6) # default vals -> rho=0.9, eps=1e-6, lr=1.0, weight_decay=0
            optimizer = torch.optim.Adam(rnet.parameters(),lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) ## default vals -> lr=0.001(or 1e-3), betas(0.9,0.999), eps=1e-8, weight_decay=0
            last_epoch = 0
            last_epoch_loss = 0
            num_chkpts = 0
            if train_afresh == False:
                for entity in os.listdir(latest_checkpoint_dir):
                    if entity[0] == ".":
                        continue
                    # Note: This will load the first checkpoint file it finds. It expects to find only one checkpoint
                    if os.path.split(entity)[1].endswith("chk"):
                        num_chkpts += 1
                        if num_chkpts > 1:
                            print("Warning: Multiple checkpoints found!")
                            break
                        abs_path = os.path.abspath(os.path.join(latest_checkpoint_dir, entity))
                        try:
                            rnet, optimizer, last_epoch, last_epoch_loss = load_from_checkpoint(abs_path ,rnet, optimizer, log_file)
                        except:
                            print(traceback.format_exc(), file=log_file)
                            print(f"Error occurred while loading model checkpoint state. Path for checkpoint: {abs_path}", file=log_file)
                            raise SystemExit
            
            print(f"\n>>> Last Training epoch was: {last_epoch} (before last checkpoint), loss was {last_epoch_loss}\n", file=log_file)
            last_epoch_loss = 0
            epoch_loss = 0
            if train_afresh:
                loss_file.truncate(0)
            for epoch in range(num_epochs):
                # Training Loop
                last_epoch_loss = epoch_loss
                epoch_loss = 0
                if last_epoch >= 0 and epoch <= last_epoch:
                    continue
                print(f"Constructing batches for {dataset_type} set at epoch {epoch}..")
                print(f"Constructing batches for {dataset_type} set at epoch {epoch}..", file=log_file)
                batches = []
                try:
                    batches = dataloader.construct_batches('training', batch_size)
                except:
                    print(f"Could not load batches from {dataset_type} set\n{traceback.format_exc()}")
                    raise SystemExit
                msg_shown = False
                for b_idx, a_batch in enumerate(batches):
                    try:
                        if not msg_shown:
                            print(f"Training epoch {epoch}..")
                            print(f"Training epoch {epoch}..", file=log_file)
                            msg_shown = True
                        passage_words_padded, ques_words_padded, answer_span, ex_idx = dataloader.map_indices_to_data_in_batch(a_batch,'training')
                        optimizer.zero_grad()
                        
                        # output is a tuple of (start_indices_probabilities_tensor, end_indices_probabilities_tensor) each of shape(batch,num_words_in_passage)
                        output = rnet(passage_words_padded, ques_words_padded)
                        start_indices_pred_prob, pred_start_indices = torch.max(output[0], dim=1)
                        end_indices_pred_prob, pred_end_indices = torch.max(output[1], dim=1)
                        loss_pos_start = F.cross_entropy(output[0], answer_span[:, 0].long().to(device))
                        loss_pos_end = F.cross_entropy(output[1], answer_span[:, 1].long().to(device))
                        # loss_pos_start = F.cross_entropy(output[0], answer_span[:, 0].long())
                        # loss_pos_end = F.cross_entropy(output[1], answer_span[:, 1].long())

                        # predicted answer span vs actual answer span
                        for i in range(output[0].shape[0]): # for every element in batch
                            # original answer span
                            o_start, o_end = answer_span[i,0], answer_span[i,1]
                            # predicted answer span
                            p_start, p_end = pred_start_indices[i], pred_end_indices[i]
                            p_start_prob, p_end_prob = start_indices_pred_prob[i], end_indices_pred_prob[i]
                            # ex is a tuple containing (qid, passsage text, question text, original answer text, reconstructed answer text, is_passage_truncated, answer_matches_original_exactly, answer_lies_beyomd_truncation_in_passage , answer_found_using_coersion(due to the effects of tokeniser having generated different tokens for answer and passage)
                            example_index = None
                            ex = None
                            try:
                                example_index = ex_idx[i]
                            except IndexError:
                                print(f"ex_idx list does not have a {i} index", file=log_file)
                                raise
                            try:
                                ex = pqa_list[example_index]
                            except IndexError:
                                print(f"pqa_list does not have a {example_index} index", file=log_file)
                                raise
                            predicted_answer = ""
                            original_answer = ""
                            predicted_answer = " ".join([token.text for token in tokenizer(ex[1])[p_start:p_end+1]])
                            if p_start > p_end:
                                predicted_answer = " ".join([token.text for token in tokenizer(ex[1])][p_start:])
                            original_answer = ex[4]
                            # Uncomment the following line to print original and predicted answers in the log along with the question text
                            print(f"Question:{ex[2]} with qid: [{ex[0]}] \nOriginal: [{original_answer}] with indices ({o_start},{o_end})\nPredicted: [{predicted_answer}] with indices ({p_start},{p_end})\n\n", file=log_file)

                        sum_loss = (loss_pos_start + loss_pos_end) / 2
                        epoch_loss += sum_loss
                        print(f"Loss epoch_{epoch} - batch{b_idx} {sum_loss}",file=log_file)
                        log_file.flush()
                        sum_loss.backward()
                        optimizer.step()
                        
                        del loss_pos_start, loss_pos_end, output, pred_start_indices, pred_end_indices, start_indices_pred_prob, end_indices_pred_prob, sum_loss
                        # if (b_idx) >= 50 and use_gpu:
                        #     torch.cuda.empty_cache()
                        del passage_words_padded, ques_words_padded, answer_span
                    except:
                        print(traceback.format_exc() ,file=log_file)
                        raise SystemExit
                print(f"\n>>> Training Loss epoch_{epoch} - {epoch_loss}\n",file=log_file)
                print(f"\n>>> Training Loss epoch_{epoch} - {epoch_loss}\n")
                print(f"{epoch},{epoch_loss}", file=loss_file)
                
                # Checkpoint
                state = {
                            'epoch': epoch,
                            'last_epoch_loss': last_epoch_loss,
                            'state_dict': rnet.state_dict(), 
                            'optimizer': optimizer.state_dict()
                }
                save_checkpoint(latest_checkpoint_dir, state, epoch, train_afresh, log_file)
                log_file.flush()
                del batches
                
                # Save model after every epoch (to compute validation errors)
                epoch_model_file = os.path.abspath(os.path.join(trained_model_dir, f"rnet_trained_epoch_{epoch}.pt"))
                print(f"\n\n>>> Epoch {epoch} training completed on full dataset, saving model at {epoch_model_file}\n---------EPOCH {epoch} COMPLETED---------\n\n", file=log_file)
                log_file.flush()
                with open(epoch_model_file,'wb+') as sfile:
                    torch.save(rnet.state_dict(), sfile)
            
            trained_model_file = os.path.abspath(os.path.join(trained_model_dir, "rnet_trained.pt"))
            print(f"\n\n>>> Training completed on full dataset, saving model at {trained_model_file}\n---------COMPLETED----------\n", file=log_file)
            log_file.flush()
            with open(trained_model_file,'wb+') as sfile:
                torch.save(rnet.state_dict(), sfile)
    except SystemExit:
        pass
    except:
        print(traceback.format_exc())