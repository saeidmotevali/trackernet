import os

sep = os.sep

DRIVE = {
    'Params': {
        'num_channels': 1,
        'num_classes': 2,
        'batch_size': 16,
        'epochs': 20,
        'learning_rate': 0.001,
        'patch_shape': (15, 15),
        'use_gpu': False,
        'distribute': False,
        'shuffle': True,
        'checkpoint_file': 'TrackNet-DRIVE.chk.tar',
        'log_frequency': 50,
        'validation_frequency': 1,
        'mode': 'train',
        'parallel_trained': False
    },
    'Dirs': {
        'image': 'data' + sep + 'DRIVE' + sep + 'mats',
        'mask': 'data' + sep + 'DRIVE' + sep + 'mask',
        'truth': 'data' + sep + 'DRIVE' + sep + 'manual',
        'logs': 'data' + sep + 'DRIVE' + sep + 'unet_logs'
    },

    'Funcs': {
        'truth_getter': lambda file_name: file_name.split('.')[0] + '_manual1.gif',
        'mask_getter': lambda file_name: file_name.split('.')[0].split('_')[-1] + '_test_mask.gif'
    }
}
