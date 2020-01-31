import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Embedding, Flatten, Reshape
from tensorflow.keras.layers import Bidirectional, GRU, LSTM
from tensorflow.keras.layers import Dense, Attention
from tensorflow.keras.layers import MaxPool1D, Conv1D
from tensorflow.keras import regularizers
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


import numpy as np

import utils


# Edited from TensorFLow official tutorial.
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, values):
    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is
    # (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(self.W(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class JEDI(tf.keras.Model):
    def __init__(self, FLAGS):
        super(JEDI, self).__init__()
        self.K = FLAGS.K
        self.L = FLAGS.L * 2
        self.max_len = FLAGS.max_len
        self.emb_dim = FLAGS.emb_dim
        self.rnn_dim = FLAGS.rnn_dim
        self.att_dim = FLAGS.att_dim
        self.hidden_dim = FLAGS.hidden_dim
        self.l2_reg = FLAGS.l2_reg
        self.build_components()

    def build_components(self):
        # K-mer embedding layer.
        self.to_embeddings = Embedding(
                5 ** self.K, self.emb_dim, mask_zero=True, input_length=self.L)
        # Site encoders.
        RNN = GRU
        self.rnn_a = Bidirectional(
                RNN(self.rnn_dim, return_sequences=True),
                backward_layer=RNN(
                    self.rnn_dim, return_sequences=True, go_backwards=True),
                merge_mode='concat')
        self.rnn_d = Bidirectional(
                RNN(self.rnn_dim, return_sequences=True),
                backward_layer=RNN(
                    self.rnn_dim, return_sequences=True, go_backwards=True),
                merge_mode='concat')
        self.seq_att_a = BahdanauAttention(self.att_dim)
        self.seq_att_d = BahdanauAttention(self.att_dim)
        # Cross attention layer.
        self.cross_attention = Attention()
        # Final attention.
        self.final_att_a = BahdanauAttention(self.att_dim)
        self.final_att_d = BahdanauAttention(self.att_dim)
        # Final prediction.
        self.concat = Concatenate()
        self.final_dense = Dense(
                self.hidden_dim,
                activation='elu',
                kernel_regularizer=regularizers.l2(self.l2_reg))
        self.predict = Dense(
                1,
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(self.l2_reg))

    def call(self, xa, xd, xlen_a, xlen_d):
        batch_size = xa.get_shape()[0]
        # Input: embeddings in a shape (batch_size, max_len * L).
        # To embeddings with a shape (batch_size, max_len * L, emb_dim).
        # Then reshape to (batch_size * max_len, L, emb_dim).
        emb_a = tf.reshape(self.to_embeddings(xa),
                (batch_size * self.max_len, self.L, self.emb_dim))
        emb_d = tf.reshape(self.to_embeddings(xd),
                (batch_size * self.max_len, self.L, self.emb_dim))

        # Encoded sites with a shape (batch_size * max_len, 2 * rnn_dim).
        # Then reshape to (batch_size, max_len, 2 * rnn_dim).
        vectors_a = tf.reshape(
                self.seq_att_a(self.rnn_a(emb_a))[0],
                (batch_size, self.max_len, self.rnn_dim * 2))
        vectors_d = tf.reshape(
                self.seq_att_d(self.rnn_d(emb_d))[0],
                (batch_size, self.max_len, self.rnn_dim * 2))

        # Cross attention between acceptor and donor sites.
        # (batch_size, max_len, 2 * rnn_dim).
        seq_mask_a = tf.sequence_mask(xlen_a, maxlen=self.max_len)
        seq_mask_d = tf.sequence_mask(xlen_d, maxlen=self.max_len)
        vectors_a_by_d = self.cross_attention(
                inputs=[vectors_d, vectors_a], mask=[seq_mask_d, seq_mask_a])

        vectors_d_by_a = self.cross_attention(
                inputs=[vectors_a, vectors_d], mask=[seq_mask_a, seq_mask_d])
        
        # Final attention to generate acceptor and donor features
        # (batch_size, 2 * rnn_dim) and then combine them.
        # (batch_size, 4 * rnn_dim)
        final_features = self.concat([
            self.final_att_a(vectors_a_by_d)[0],
            self.final_att_d(vectors_d_by_a)[0]])
        
        # Final MLP for prediction.
        final_features = self.final_dense(final_features)
        pred = self.predict(final_features)
        return pred

