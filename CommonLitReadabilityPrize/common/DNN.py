import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import string
import sys
import time

from pathlib import Path
import yaml

from tqdm import tqdm
from functools import partial
from nltk.corpus import words, stopwords

import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model, load_model, model_from_yaml
from tensorflow.keras.layers import Conv1D, \
    Flatten, \
    GlobalMaxPooling1D, \
    TimeDistributed, \
    MaxPooling1D, \
    Dense, \
    Activation, \
    ReLU, \
    LSTM, \
    GRU, \
    SpatialDropout1D, \
    Dropout, \
    Bidirectional, \
    Embedding, \
    Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Layer, InputSpec
from datetime import datetime

random.seed(0)
np.random.seed(0)

sys.path.insert(0, '../src/')

from train_config import config

punctuation = string.punctuation.replace("@", "").replace("+", "").replace("-", "").replace("_", "")
stop_words = set(stopwords.words('english'))


def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        # print(f"\nFunc {func.__name__} took {(t2 - t1)}")
        return res

    return wrapper


class ResidualBlock1D(Layer):
    def __init__(self, channels_in, kernel, **kwargs):
        super(ResidualBlock1D, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel

        self.conv1 = Conv1D(self.channels_in,
                            self.kernel,
                            padding='same',
                            activation='relu')
        self.conv2 = Conv1D(self.channels_in,
                            self.kernel,
                            padding='same')
        self.activation = Activation('relu')

    def call(self, x):
        y = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = Add()([x, y])
        x = self.activation(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'channels_in': self.channels_in,
            'kernel': self.kernel,
        })
        return config


class DNN:
    def __init__(
            self,
            backbone='LSTM',  # New option TEST_CLIT
            charlevel=False,
            use_glove=True,
            preprocess_data=False,
            use_sensitive_tokens=False,
            logger=None,
            batch_size=64,
            max_vocab=10000,
            max_len=8192,
            embedding_mat_columns=50,
            use_multilabel=False,
            epochs=200
    ):

        self.backbone = backbone
        self.charlevel = charlevel
        self.use_glove = use_glove
        self.preprocess_data = preprocess_data

        self.OOV_TOK = '<OOV>'
        self.PADDING_TYPE = 'post'
        self.TRUNCATE_TYPE = 'pre'  # 'post'
        self.batch_size = batch_size
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embedding_mat_columns = embedding_mat_columns

        self.__model = None
        self.res = list()
        self.is_trained = False

        self.logger = logger
        self.epochs = epochs

        self.augment = False
        self.multilabel = use_multilabel

        if self.charlevel and self.use_glove:
            if logger:
                logger.warning('charlevel and use_glove both set to true. use_glove will be ignored.')

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value

    @staticmethod
    def clean_text(txt):
        return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()

    def preprocess_texts(self, texts, tokens=[]):
        preprocessed_texts = []
        additional_tokens = []

        for text in texts:
            if self.preprocess_data:
                text = self.clean_text(text)

            preprocessed_texts.append(text)

        return np.array(preprocessed_texts), tokens, np.array(additional_tokens)

    def fit_tokenizer(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, tuple):
            texts = list(texts)
        elif not isinstance(texts, (list, pd.core.series.Series, np.ndarray)):
            raise ValueError("The text must be a list of strings, a list of lists containing strings or a string")

        self.tokenizer.fit_on_texts(texts)

    def sequence_padding(self, sequences):
        seqs = pad_sequences(sequences, maxlen=self.max_len, padding=self.PADDING_TYPE, truncating=self.TRUNCATE_TYPE)
        return seqs

    def load_glove(self, additional_tokens=[]):
        embeddings_index = {}
        glove_path = f'../../data/glove.6B.{self.embedding_mat_columns}d.txt'  # Todo change it if running from MAIN
        f = open(glove_path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        # print(len(embeddings_index))

        # embedding_matrix = np.zeros((len(embeddings_index) + 2 + len(additional_tokens), self.embedding_mat_columns))
        embedding_matrix = np.random.randn(len(embeddings_index) + 2 + len(additional_tokens),
                                           self.embedding_mat_columns)
        w2i = {}
        for i, (word, embs) in enumerate(embeddings_index.items()):
            embedding_matrix[i] = embs
            w2i[word] = i
        i += 1
        w2i['<OOV>'] = i

        if self.logger:
            self.logger.info(f'load_glove: {embedding_matrix.shape}, {len(w2i.keys())}')
        else:
            print('load_glove', embedding_matrix.shape, len(w2i.keys()))

        return embeddings_index, w2i, embedding_matrix

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_model(self, vocab_size, embedding_matrix=None):

        model = Sequential()
        model.add(Embedding(
            vocab_size,
            self.embedding_mat_columns,
            embeddings_initializer=tf.keras.initializers.Constant(
                embedding_matrix) if embedding_matrix is not None else None
        ))

        if self.backbone == 'LSTM':
            model.add(Bidirectional(LSTM(self.embedding_mat_columns)))
        elif self.backbone == 'GRU':
            model.add(Bidirectional(GRU(self.embedding_mat_columns)))
        elif self.backbone == 'CONV':
            model.add(Conv1D(512, 3, activation='relu'))
            model.add(ResidualBlock1D(512, 3))
            model.add(MaxPooling1D())
            model.add(Conv1D(256, 3, activation='relu'))
            model.add(ResidualBlock1D(256, 3))
            model.add(MaxPooling1D())
            model.add(Conv1D(128, 3, activation='relu'))
            model.add(ResidualBlock1D(128, 3))
            model.add(GlobalMaxPooling1D())
        elif self.backbone == 'DEMO':
            model.add(Conv1D(256, 3, activation='relu'))
            model.add(GlobalMaxPooling1D())
        elif self.backbone == 'TEST_CLIT':
            model.add(Conv1D(64, 5, padding='valid', kernel_initializer='normal', activation='relu'))
            model.add(MaxPooling1D(2))
            model.add(Conv1D(128, 5, padding='valid', kernel_initializer='normal', activation='relu'))
            model.add(MaxPooling1D(2))
            model.add(Conv1D(256, 5, padding='valid', kernel_initializer='normal', activation='relu'))
            model.add(Conv1D(512, 5, padding='valid', kernel_initializer='normal', activation='relu'))
            model.add(Conv1D(1024, 5, padding='valid', kernel_initializer='normal', activation='relu'))
            model.add(Conv1D(2048, 5, padding='valid', kernel_initializer='normal', activation='relu'))
            model.add(Conv1D(4098, 5, padding='valid', kernel_initializer='normal', activation='relu'))
            model.add(Conv1D(8196, 5, padding='valid', kernel_initializer='normal', activation='relu'))
            model.add(GlobalMaxPooling1D())
            model.add(Dense(120, kernel_initializer='normal', activation='relu'))
            model.add(Dense(240, kernel_initializer='normal', activation='relu'))
            model.add(Dense(480, kernel_initializer='normal', activation='relu'))
            model.add(Dense(980, kernel_initializer='normal', activation='relu'))
            model.add(Dense(1500, kernel_initializer='normal', activation='relu'))
        else:
            raise NotImplementedError

        model.add(Dense(1))

        return model

    def _augment(self, X_train, y_train, n_samples=30000):
        X_train_aug = []
        y_train_aug = []

        for _ in tqdm(range(n_samples)):
            y_new = random.randint(0, 2)

            idx = np.arange(len(y_train))
            idx = idx[y_train <= y_new]

            random.shuffle(idx)
            idx = idx[:random.randint(10, 50)]

            x_new = '\n'.join(X_train[idx])
            y_new = np.max(y_train[idx])

            X_train_aug.append(x_new)
            y_train_aug.append(y_new)

        return np.array(X_train_aug), np.array(y_train_aug)

    def _augment_multilabel(self, X_train, y_train, n_samples=30000):
        X_train_aug = []
        y_train_aug = []

        for i in tqdm(range(n_samples)):
            y_new = [random.randint(0, 1), random.randint(0, 1)]

            idx = np.arange(len(y_train))
            idx = idx[(y_train == 0) | (y_train == y_new[0] * 1) | (y_train == y_new[1] * 2)]

            random.shuffle(idx)
            idx = idx[:random.randint(10, 50)]

            x_new = '\n'.join(X_train[idx])
            y_train_idx = y_train[idx]
            y_new = np.array([int(np.any(y_train_idx == 1)), int(np.any(y_train_idx == 2))])

            X_train_aug.append(x_new)
            y_train_aug.append(y_new)

            # if i >= 4000:
            #     break

        return np.array(X_train_aug), np.array(y_train_aug)

    def fit(self, X_train, y_train, out_path):
        X, y = X_train, y_train

        if not Path(out_path).exists():
            os.makedirs(out_path)

        if self.augment:
            self.logger.info(f'Augmenting. Initial number of samples: {len(X)}')
            if self.multilabel:
                if os.path.isfile('train_temp_multilabel.csv'):
                    df = pd.read_csv('train_temp_multilabel.csv')
                    X = df['X'].values
                    y = np.hstack([df['y_0'].values.reshape(-1, 1), df['y_1'].values.reshape(-1, 1)])
                else:
                    X_train_aug, y_train_aug = self._augment_multilabel(X, y)
                    X = X_train_aug
                    y = y_train_aug

                    df = pd.DataFrame()
                    df['X'] = X
                    df['y_0'] = y[:, 0]
                    df['y_1'] = y[:, 1]
                    df.to_csv('train_temp_multilabel.csv', index=False)
                self.logger.info(f'Augmentation done. Number of samples: {len(X)}')

                # lens = [len(x) for x in X]
                # print(np.min(lens), np.max(lens), np.mean(lens))

                labels_encoded = y

            else:
                if os.path.isfile('train_temp.csv'):
                    df = pd.read_csv('train_temp.csv')
                    X = df['X'].values
                    y = df['y'].values
                else:
                    X_train_aug, y_train_aug = self._augment(X, y)
                    X = X_train_aug
                    y = y_train_aug

                    df = pd.DataFrame()
                    df['X'] = X
                    df['y'] = y
                    df.to_csv('train_temp.csv', index=False)
                self.logger.info(f'Augmentation done. Number of samples: {len(X)}')

                # lens = [len(x) for x in X]
                # print(np.min(lens), np.max(lens), np.mean(lens))

                labels_encoded = pd.get_dummies(y).values

        else:
            # lens = [len(x) for x in X]
            # print(np.min(lens), np.max(lens), np.mean(lens))

            labels_encoded = pd.get_dummies(y).values

        additional_tokens = []

        X, self.tokens, additional_tokens = self.preprocess_texts(X)

        # print(X.shape, labels_encoded.shape)
        # print(np.mean(labels_encoded, axis=0))

        embedding_matrix = None
        if self.charlevel:
            self.tokenizer = Tokenizer(oov_token=self.OOV_TOK, filters='', lower=False, char_level=True)
            self.fit_tokenizer(texts=X)
        else:
            self.tokenizer = Tokenizer(oov_token=self.OOV_TOK, lower=True,
                                       char_level=False)  # filters='<', lower=False,

            if self.use_glove:
                _, w2i, embedding_matrix = self.load_glove(additional_tokens=additional_tokens)
                self.tokenizer.word_index = w2i
            else:
                self.fit_tokenizer(texts=X)

        # sequences = np.array(self.tokenizer.texts_to_sequences(X))
        sequences = self.tokenizer.texts_to_sequences(X)

        X = self.sequence_padding(sequences)

        self.model = self.get_model(vocab_size=(len(self.tokenizer.word_index) + 1), embedding_matrix=embedding_matrix)

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.model.compile(loss='mean_squared_error', optimizer=opt, metrics=[tf.metrics.MeanSquaredError()])

        self.model.summary()

        es = EarlyStopping(
            monitor='val_mean_squared_error',
            mode='max',
            patience=20,
            verbose=1
        )
        lr_sch = ReduceLROnPlateau(
            monitor='val_mean_squared_error',
            mode='max',
            factor=0.1,
            patience=10,
            verbose=1,
            min_delta=0.001,
            cooldown=0,
            min_lr=1e-6,
        )
        ckpt = ModelCheckpoint(
            os.path.join(out_path, 'model.h5'),
            monitor='val_mean_squared_error',
            mode='max',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            save_freq='epoch'
        )

        # print(X.shape, labels_encoded.shape)
        try:
            # print(self.batch_size)
            self.model.fit(X, labels_encoded, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, verbose=1,
                           callbacks=[es, lr_sch, ckpt])
        except KeyboardInterrupt:
            print('Got KeyboardInterrupt. Stopping.')

        self.is_trained = True

    def predict_proba(self, X):
        assert self.is_trained, 'Model should be trained before inference.'
        if isinstance(X, str):
            X = [X]

        if self.preprocess_texts:
            X, _, _ = self.preprocess_texts(X, tokens=self.tokens)

        sequences = self.tokenizer.texts_to_sequences(X)
        padded = self.sequence_padding(sequences)
        preds = self.model.predict(padded)

        return preds

    def predict(self, X, return_proba=False):
        assert self.is_trained, 'Model should be trained before inference.'

        proba = self.predict_proba(X)
        preds = np.argmax(proba, axis=1)

        if return_proba:
            return preds, proba
        else:
            return preds

    def save(self, path):
        if self.is_trained:

            output_dir = Path(path)
            if not output_dir.exists():
                Path.mkdir(output_dir, parents=True, exist_ok=True)

            # serialize model to YAML
            model_yaml = self.model.to_yaml()
            with open(output_dir / 'nn_model_config.yaml', 'w') as file:
                file.write(model_yaml)
            # serialize weights to HDF5
            self.model.save_weights(output_dir / "model.h5")

            with open(output_dir / 'tokenizer.pkl', 'wb') as file:
                pickle.dump(self.tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

            self.logger.info(f'Saved model to {output_dir}')
        else:
            self.logger.warning('Cannot save the model. Train it first.')

    def load(self, path):
        output_dir = Path(path)

        with open(output_dir / 'nn_model_config.yaml') as file:
            model_config = file.read()

        self.model = model_from_yaml(model_config)
        self.model.load_weights(output_dir / "model.h5")

        with open(output_dir / 'tokenizer.pkl', 'rb') as file:
            self.tokenizer = pickle.load(file)

        self.is_trained = True
