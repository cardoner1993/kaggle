#!/usr/bin/env python
import datetime

config = {}

config['epochs'] = 500 # Training iterations

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
    'bert-base-uncased': {
        'model': 'BertClassifier', 'architecture': 'BERT'
    },
    'dnn_char_lstm': {
        'model': 'DNN', 'architecture': 'DNN', 'parameters': {
            'backbone': 'LSTM',
            'charlevel': True,
            'max_len': 16384,
            'epochs': 200
            }
    },
    'dnn_char_conv': {
        'model': 'DNN', 'architecture': 'DNN', 'parameters': {
            'backbone': 'CONV',
            'charlevel': True,
            'batch_size': 32,
            'max_len': 16384,
            'epochs': 200
            }
    },
    'dnn_lstm': {
        'model': 'DNN', 'architecture': 'DNN', 'parameters': {
            'backbone': 'LSTM',
            'epochs': 200
        }
    },
    'dnn_conv': {
        'model': 'DNN', 'architecture': 'DNN', 'parameters': {
            'backbone': 'CONV',
            'epochs': 200
        }
    },
    'dnn_demo': {
        'model': 'DNN', 'architecture': 'DNN', 'parameters': {
            'backbone': 'DEMO',
            'batch_size': 16,
            'epochs': 200
            }
    },
    'dnn_clit': {
        'model': 'DNN', 'architecture': 'DNN', 'parameters': {
            'backbone': 'TEST_CLIT',
            'epochs': 200
        }
    },
}

