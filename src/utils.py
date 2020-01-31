from absl import flags
from absl import logging
import tensorflow as tf
import numpy as np
import random
import sys
try:
    import ujson as json
except:
    import json


def handle_flags():
    flags.DEFINE_string("tflog",
            '3', "The setting for TF_CPP_MIN_LOG_LEVEL (default: 3)")
    # Data configuration.
    flags.DEFINE_string('config',
            'config.yml', 'configure file (default: config.yml)')
    flags.DEFINE_integer('cv', 0, 'Fold for cross-validation (default: 0)')
    flags.DEFINE_integer('K', 3, 'K for k-mers (default: 3)')
    flags.DEFINE_integer('L', 4, 'Length for junction sites (default: 4)')

    # Model parameters.
    flags.DEFINE_integer('emb_dim',
            128, 'Dimensionality for k-mers (default: 129)')
    flags.DEFINE_integer('rnn_dim',
            128, 'Dimensionality for RNN layers (default: 128)')
    flags.DEFINE_integer('att_dim',
            16, 'Dimensionality for attention layers (default: 16)')
    flags.DEFINE_integer('hidden_dim',
            128, 'Dimensionality for hidden layers (default: 128)')
    flags.DEFINE_integer('max_len',
            128, 'Max site number for acceptors/donors (default: 128)')
    # Training parameters.
    flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    flags.DEFINE_integer("num_epochs",
            10, "Number of training epochs (default: 10)")
    flags.DEFINE_integer('random_seed',
            252, 'Random seeds for reproducibility (default: 252)')
    flags.DEFINE_float('learning_rate',
            1e-3, 'Learning rate while training (default: 1e-3)')
    flags.DEFINE_float('l2_reg',
            1e-3, 'L2 regularization lambda (default: 1e-3)')
    FLAGS = flags.FLAGS


def limit_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
          print(e)
          return False
    return True


class Data:
    def __init__(self, file_name, FLAGS):
        self.L = FLAGS.L
        self.max_len = FLAGS.max_len
        self.batch_size = FLAGS.batch_size
        self.records = []
        flatten = lambda l: [item for sublist in l for item in sublist]
        with open(file_name, 'r') as fp:
            for line in fp:
                data = json.loads(line)
                assert(len(data['acceptors'][0]) == self.L)
                assert(len(data['donors'][0])  == self.L)
                self.records.append({
                    'acceptors': flatten(data['acceptors']),
                    'donors': flatten(data['donors']),
                    'length_a': len(data['acceptors']),
                    'length_d': len(data['donors']),
                    'label': data['label']})
        logging.info('Loaded {} records from {}.'.format(len(self.records),
            file_name))

    def pad(self, x):
        y = np.zeros(self.L * self.max_len, dtype=np.int32)
        RL = min(len(x), self.L * self.max_len)
        y[:RL] = x[:RL]
        return y

    def batch_iter(self, is_random=True):
        if is_random:
            random.shuffle(self.records)
        cur_a, cur_d, cur_len_a, cur_len_d, cur_lbl = [], [], [], [], []

        cur_cnt = 0
        for data in self.records:
            cur_lbl.append([data['label']])
            cur_a.append(self.pad(data['acceptors']))
            cur_d.append(self.pad(data['donors']))
            cur_len_a.append(min(data['length_a'], self.max_len))
            cur_len_d.append(min(data['length_d'], self.max_len))
            cur_cnt += 1
            if cur_cnt == self.batch_size:
                yield {
                        'label': np.array(cur_lbl),
                        'acceptors': np.array(cur_a, dtype=np.int32),
                        'donors': np.array(cur_d, dtype=np.int32),
                        'length_a': np.array(cur_len_a, dtype=np.int32),
                        'length_d': np.array(cur_len_d, dtype=np.int32)}
                cur_cnt = 0
                cur_a, cur_d, cur_len_a, cur_len_d, cur_lbl = [], [], [], [], []
        yield {
                'label': np.array(cur_lbl),
                'acceptors': np.array(cur_a, dtype=np.int32),
                'donors': np.array(cur_d, dtype=np.int32),
                'length_a': np.array(cur_len_a, dtype=np.int32),
                'length_d': np.array(cur_len_d, dtype=np.int32)}

