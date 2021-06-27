import argparse
import datetime
import glob
import json
import logging
import numpy as np
import sys
import time
import traceback
import pandas as pd
from functools import partial
import os

from tqdm import tqdm
from termcolor import colored
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
sys.path.append('../')
sys.path.append('..')

from src.train_config import config


def train_models(config=config, logger=None, use_bert=False):
    
    data_train = pd.read_csv(config['train_data'])
    data_train.reset_index(drop=True)

    if use_bert:
        from common.Bert_Finetune import BertClassifier
        out_path = os.path.join(config['output_dir'], config['model_name'], 'v1')
        model = BertClassifier(logger)
        model.fit(data_train["excerpt"], data_train["target"])
    else:
        from common.DNN import DNN
        backbone = 'DEMO'
        model_name = f"{backbone}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_path = os.path.join(config['output_dir'], model_name)
        model = DNN(logger=logger, backbone=backbone, epochs=2)
        model.fit(data_train["excerpt"], data_train["target"], out_path=out_path)

    model.save(out_path)


if __name__ == '__main__':
    # Logging
    logger = logging.getLogger('train_models')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(colored('[%(asctime)s]', 'magenta') +
                                  colored('[%(levelname)s] ',
                                          'blue') + '%(message)s',
                                  '%Y-%m-%d %H:%M:%S')

    logging_file_handler = logging.FileHandler(
        f'train_models_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging_file_handler.setLevel(logging.DEBUG)
    logging_file_handler.setFormatter(formatter)
    logger.addHandler(logging_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f'Starting model training PID: ' +
                colored(f'{os.getpid()}', 'green'))
    train_start_time = time.time()


    logger.info('Working directory: ' + colored(f'{config["output_dir"]}', 'green'))

    train_models(config=config, logger=logger)

    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    logger.info(f'Train time: {train_time:.2} s.')
    logger.info(f'Done model training PID: ' +
                colored(f'{os.getpid()}', 'green'))
