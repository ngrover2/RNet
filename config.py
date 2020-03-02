import os

base_config = {
    # Glove Word Embeddings file path. Provide path for the word embeddings file in one of the following ways
    # 1 full dir path and file name separately
    "use_gpu": True,
    "glove_word_embed_dir" : './glove_embeddings/glove_words',
    "word_embed_file_name" : 'glove.6B.words.300d.txt',
    # OR 2. full path including file name
    "word_embed_file_path" : './glove_embeddings/glove_words/glove.6B.words.300d.txt',
    
    "log_dir" : os.path.join(os.path.abspath(''), 'logs'),
    "vocab_dicts_log_file_name" : 'id2word_word2id_dict_creation.log',
    "load_training_set_log_file_name" : 'load_training_set.log',
    "load_dev_set_log_file_name" : 'load_dev_set.log',

    # file path for logs about creating word2index and index2word dictionaries
    "save_dict_dir" : 'embedding_dicts',
    "id2word_dict_file_name" : 'id2word_dict',
    "word2id_dict_file_name" : 'word2id_dict',
    
    # # #
    # Use one of the folowwing ways to give the path for training and dev datasets
    # Method 1
    # dataset full file path. Use this to give a direct path to the file where the dev dataset exists
    "training_dataset_file" : './dataset_squad/training_set/squad_train_v2.0.json',
    "dev_dataset_file" : './dataset_squad/dev_set/squad_dev_v2.0.json',
    # OR give dev dataset sub dir name within the 'dataset_dir' and the file name separately
    # Method 2. 
    # dir names and file path names where the SQuAD dataset exists
    "dataset_dir" : os.path.abspath(os.path.join(os.path.abspath(''), 'dataset_squad')),
    "dataset_training_subdir" : 'training_set',
    "dataset_training_file_name" : 'squad_train_v2.0.json',
    "dataset_dev_subdir" : 'dev_set',
    "dataset_dev_file_name" : 'squad_dev_v2.0.json',
    # # #

    "human_readable_training_dataset_dir" : "dataset_readable/training",
    "human_readable_dev_dataset_dir" : "dataset_readable/dev",
    "max_passage_len": 300,
    "word_embedding_dim" : 300,
    "training_batch_size" : 10,
    "dev_batch_size" : 10,
    "train_n_examples" : 30,
    "dev_n_examples" : 1000,
    "latest_training_checkpoint_dir" : "./latest_checkpoint/training",
    "latest_dev_checkpoint_dir" : "./latest_checkpoint/dev",
    "saved_object_states_dir" : "./saved_objects",
    "list_passage_question_answer_tuples_dir" : "./saved_objects/list_PQAs.pkl",
    "train_afresh": True,
    "num_epochs": 200,
    "trained_model_dir": "./trained_model",
    "pred_dir": "./predictions",
    # char_embed_file : 'glove.6B.chars.300d.txt',
}