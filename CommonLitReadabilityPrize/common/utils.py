import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import gc
import tensorflow as tf
from tensorflow.python.keras import backend as K


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    gc.collect()
    torch.cuda.empty_cache()
    # device = cuda.get_current_device()
    # device.reset()

    # current_device = cuda.get_current_device().id
    # cuda.select_device(current_device)
    # cuda.close()
    # cuda.select_device(current_device)

    print("GPU Usage after emptying the cache")
    gpu_usage()


def config_gpu_growth():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)

    return sess


def close_sess_keras(sess):
    sess.close()
    tf.keras.backend.clear_session()