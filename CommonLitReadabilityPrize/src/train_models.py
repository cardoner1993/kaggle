import argparse
import datetime
import glob
import json
import logging
import multiprocessing

import numpy as np
import sys
import time
import traceback
import pandas as pd
from functools import partial
import os
import fire

from tqdm import tqdm
from termcolor import colored
from pathlib import Path
import torch

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.append('../common')

from train_config import config
from utils import free_gpu_cache, config_gpu_growth, close_sess_keras


# Todo Not used because of GPU errors
def train_models(config=config, logger=None):
    data_train = pd.read_csv(config['train_data'])
    data_train.reset_index(drop=True)

    for model_type in config['models']:
        model_config = config['models'][model_type]
        logger.info(f"Running model {model_type}")

        if model_type == 'BERT':

            from Bert_Finetune import BertClassifier, run_pytorch

            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                raise ValueError("Not Valid spawn method")

            for model in model_config:
                # free_gpu_cache()
                out_path = os.path.join(config['output_dir'], 'BERT', model, 'v1')
                # option base. Clean model and optimizer with release memory after training
                # model = BertClassifier(logger=logger, **model_config[model]['parameters'])
                # model.fit(data_train["excerpt"], data_train["target"])

                # model.save(out_path)
                # model.release_memory()

                # option 1: execute code with extra process
                p = multiprocessing.Process(target=run_pytorch, args=(data_train["excerpt"],
                                                                      data_train["target"],
                                                                      model_config[model]['parameters'],
                                                                      out_path,))
                p.start()
                p.join()

        elif model_type == 'DNN':
            from DNN import DNN, run_tensorflow

            for model in model_config:
                base_path = os.path.join(config['output_dir'], 'DNN', model,
                                         datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
                os.makedirs(base_path, exist_ok=True)
                # model = DNN(logger=logger, **model_config[model]['parameters'])
                # model.fit(data_train["excerpt"], data_train["target"], out_path=base_path)
                #
                # model.save(base_path)
                # option 1: execute code with extra process
                p = multiprocessing.Process(target=run_tensorflow, args=(data_train["excerpt"],
                                                                         data_train["target"],
                                                                         model_config[model]['parameters'],
                                                                         base_path,))
                p.start()
                p.join()
                # run_tensorflow(data_train["excerpt"], data_train["target"], model_config[model]['parameters'], base_path)
        else:
            logger.error("The option defined in config is not implemented yet. Continue")


def train_model(model_type, model_name=None, logger=None):
    # Model name must be a string with either BERT or DNN and backbone the parameter to use in case of DNN module used
    data_train = pd.read_csv(config['train_data'])
    data_train.reset_index(drop=True)

    if model_type == 'BERT':

        from Bert_Finetune import BertClassifier

        model_config = config['models'][model_type][model_name]
        out_path = os.path.join(config['output_dir'], model_name, 'v1')
        model = BertClassifier(logger=logger, **model_config['parameters'])
        model.fit(data_train["excerpt"], data_train["target"])

        model.save(out_path)

    elif model_type == 'DNN':

        from DNN import DNN

        model_config = config['models'][model_type][model_name]
        base_path = os.path.join(config['output_dir'], model_type)
        os.makedirs(base_path, exist_ok=True)

        model_path = f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_path = os.path.join(base_path, model_path)

        model = DNN(logger=logger, **model_config['parameters'])
        model.fit(data_train["excerpt"], data_train["target"], out_path=out_path)

        model.save(out_path)

    else:
        logger.error("The option defined in config is not implemented yet. Continue")


if __name__ == '__main__':
    # Logging
    logger = logging.getLogger('train_models')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(colored('[%(asctime)s]', 'magenta') +
                                  colored('[%(levelname)s] ',
                                          'blue') + '%(message)s',
                                  '%Y-%m-%d %H:%M:%S')

    # logging_file_handler = logging.FileHandler(
    #     f'train_models_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    # logging_file_handler.setLevel(logging.DEBUG)
    # logging_file_handler.setFormatter(formatter)
    # logger.addHandler(logging_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f'Starting model training PID: ' +
                colored(f'{os.getpid()}', 'green'))
    train_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', required=True, help='Model type. EX: BERT or DNN (mayusc)')
    parser.add_argument('-n', '--name', required=True, help='DNN or BERT name. Options are: '
                                                            'bert-base-uncased '
                                                            'dnn_char_lstm, dnn_char_conv, dnn_lstm,'
                                                            'dnn_conv, dnn_demo, dnn_clit')

    args = parser.parse_args()

    assert args.type.lower() in args.name, "If using BERT as type pass as name a bert architecture and viceversa"

    logger.info('Working directory: ' + colored(f'{config["output_dir"]}', 'green'))

    # train_models(config=config, logger=logger)
    # train_model(model_type=args.type, model_name=args.name, logger=logger)
    train_models(config=config, logger=logger)

    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    logger.info(f'Train time: {train_time:.2} s.')
    logger.info(f'Done model training PID: ' +
                colored(f'{os.getpid()}', 'green'))
