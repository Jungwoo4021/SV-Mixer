import os
import itertools

def get_args():
    args = {
        # expeirment info
        'project'                   : 'SV_Mixer',
        'name'                      : 'test_code',
        'tags'                      : ['Release'],
        'description'               : '',
        'result'                    : './', # final destination: result + project + name

        #FIXME
        #---------------------------------
        # dataset path 
        'path_vox2_train'           : '/data/list/vox2_train_samples.txt',
        'path_vox_O_trials'         : '/data/list/vox2_testO_trials.txt',
        'path_vox_E_trials'         : '/data/list/vox2_testE_trials.txt',
        'path_vox_H_trials'         : '/data/list/vox2_testH_trials.txt',
        'path_vcmix_trials'         : '/data/list/vcmix_test.txt',
        'path_voxsrc_trials'        : '/data/list/voxsrc_test.txt',
        'path_voices_dev_trials'    : '/data/list/voices_dev.txt',
        'path_voices_eval_trials'   : '/data/list/voices_eval.txt',

        # model version
        'pretrained_version'        : 'Large', # or 'Small'
        #---------------------------------

        'path_pretrined_svmixer'    : None,
        'path_pretrined_classifier' : None,
        
        # experiment
        'batch_size'            : 128,
        
        # model
        'hidden_size'           : 1024,
        'seq_len'               : 149,
        'channel'               : 512,
        'num_hidden_layers'     : 5,
        'embedding_size'        : 192,

        # data processing
        'num_seg'               : 10,
        'crop_size'             : 16000 * 3, # 3sec
    }

    if args['pretrained_version'] is 'Largs':
        args['num_hidden_layers'] = 17
        args['path_pretrined_svmixer'] = 'https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_17layer_eer0.78_student.pt'
        args['path_pretrined_classifier'] = 'https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_17layer_eer0.78_classifier.pt'
    elif args['pretrained_version'] is 'Small':
        args['num_hidden_layers'] = 5
        args['path_pretrined_svmixer'] = 'https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_5layer_eer0.91_student_model.pt'
        args['path_pretrined_classifier'] = 'https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_5layer_eer0.91_classifier.pt'

    return args