#!/usr/bin/env python
import datetime

config = {}

config['epochs'] = 500  # Training iterations

config['output_dir'] = f'../models/'  # Directory where all results are saved

config['train_size'] = 0.8

config['train_data'] = "../../data/train.csv"
config['test_data'] = "../../data/test.csv"
config['submission'] = "../../data/sample_submission.csv"

# "bert-base-uncased" structure
config['MAX_LEN'] = 512
config['lr'] = 2e-5
config['eps'] = 1e-8
config['batch_size'] = 10

config['models'] = {
    'BERT': {
        'bert-base-uncased': {
            'parameters': {
                'model_name': 'bert-base-uncased',
                'batch_size': 10,
                'max_len': 512,
                'lr': 2e-5,
                'eps': 1e-8,
                'epochs': 2
            }
        },
        'roberta-base': {
            'parameters': {
                'model_name': 'roberta-base',
                'batch_size': 10,
                'max_len': 512,
                'lr': 2e-5,
                'eps': 1e-8,
                'epochs': 2
            }
        }

    },
    'DNN': {
        'dnn_char_lstm': {
            'parameters': {
                'backbone': 'LSTM',
                'charlevel': True,
                'max_len': 16384,
                'epochs': 2
            }
        },
        'dnn_char_conv': {
            'parameters': {
                'backbone': 'CONV',
                'charlevel': True,
                'batch_size': 32,
                'max_len': 16384,
                'epochs': 2
            }
        },
        'dnn_lstm': {
            'parameters': {
                'backbone': 'LSTM',
                'epochs': 2
            }
        },
        'dnn_conv': {
            'parameters': {
                'backbone': 'CONV',
                'epochs': 2
            }
        },
        'dnn_demo': {
            'parameters': {
                'backbone': 'DEMO',
                'batch_size': 16,
                'epochs': 2
            }
        },
        'dnn_clit': {
            'parameters': {
                'backbone': 'TEST_CLIT',
                'epochs': 2
            }
        },
    }
}

# epochs where 2 put 200 and where 5 put 500
