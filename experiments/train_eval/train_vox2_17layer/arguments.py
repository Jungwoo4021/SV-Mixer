import os
import itertools

def get_args():
    args = {
        # expeirment info
        'project'               : 'IRLAB/LightweightStudent2025',
        'name'                  : 'train_vox2_17layer',
        'tags'                  : ['Propposed'],
        'description'           : '',
        'result'                : '/results', # final destination: result + project + name
        'neptune_token'         : '',
        'seed'                  : 4221,

        # dataset path
        'path_vox_train'       : '/data/list/vox2_train_samples.txt',
        'path_vox_O_trials'     : '/data/list/vox2_testO_trials.txt',
        'path_musan'            : '/data/list/musan.txt',
        'path_rir'              : '/data/list/rir_noises.txt',

        # huggingface model
        'huggingface_url'       : 'microsoft/wavlm-large',
        
        # experiment
        'batch_size'            : 128,
        'eval_interval_steps'   : 2000,
        'rise_steps'            : 2000,
        'warmup_steps'          : 30000,
        'train_steps'           : 30000 * 9,
        'total_train_steps'     : 304000,
        
        # model
        'hidden_size'           : 1024,
        'teacher_hidden_size'   : 1024,
        'seq_len'               : 149,
        'channel'               : 512,
        'num_hidden_layers'     : 17,
        'embedding_size'        : 192,
        'aam_margin'            : 0.2,
        'aam_scale'             : 30,
        'topk_panalty'          : (5, 0.1),
        'cos_lambda'            : 1, 
        'target_layer_idx'      : [8, 16, 24],

        # data processing
        'num_seg'               : 10,
        'crop_size'             : 16000 * 3, # 3sec
        'augment_probability'   : 0.8,
    
        # learning rate
        'warmup_lr_max'         : 1e-3,
        'warmup_lr_min'         : 1e-4,
        'lr_max'                : 1e-1,
        'lr_min'                : 1e-4,
        'momentum'              : 0.9,
        'weight_decay'          : 2e-5,
    }
    
    return args