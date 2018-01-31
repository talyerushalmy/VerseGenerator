from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv

# GPU usage settings
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse
from RNN_utils import *

# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default = './data/data.txt')
ap.add_argument('-batch_size', type = int, default = 300)
ap.add_argument('-layer_num', type = int, default = 2)
ap.add_argument('-seq_length', type = int, default = 50)
ap.add_argument('-hidden_dim', type = int, default = 500)
ap.add_argument('-generate_length', type = int, default = 500)
ap.add_argument('-nb_epoch', type = int, default = 20)
ap.add_argument('-mode', default = 'train')
ap.add_argument('-weights', default = '')
ap.add_argument('-sample_interval', type = int, default = 5)
ap.add_argument('-save_interval', type = int, default = 5)
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
SAMPLE_INTERVAL = args['sample_interval']
SAVE_INTERVAL = args['save_interval']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

# Creating training data
X, y, VOCAB_SIZE, ix_to_char = load_data(DATA_DIR, SEQ_LENGTH)

# Creating and compiling the Network
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape = (None, VOCAB_SIZE), return_sequences = True))
for i in range(LAYER_NUM - 1):
  model.add(LSTM(HIDDEN_DIM, return_sequences = True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop")

'''
# Generate some sample before training to know how bad it is!
print('[*] Generating sample...')
generate_text(model, args['generate_length'], VOCAB_SIZE, ix_to_char)
'''

if not WEIGHTS == '':
  model.load_weights(WEIGHTS)
  nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
else:
  nb_epoch = 0

# Training if there is no trained weights specified
if args['mode'] == 'train' or WEIGHTS == '':
  while True:
    print('\n\n[*] Epoch: {}\n'.format(nb_epoch))
    model.fit(X, y, batch_size = BATCH_SIZE, verbose = 1, nb_epoch = 1)
    nb_epoch += 1
    # Generate 2 samples every SAMPLE_INTERVAL epochs
    if nb_epoch % SAMPLE_INTERVAL == 0:
      print('\n\n' + '#' * 80 + '\nRandom generation 1:')
      generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
      print('\n\n' + '#' * 80 + '\nRandom generation 2:')
      generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
      print('\n\n' + '#' * 80 + '\n')
    # Save every SAVE_INTERVAL epochs
    if nb_epoch % SAVE_INTERVAL == 0:
      print('\n[*] Saving model...')
      model.save('./model_checkpoints/checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch))
      print('[+] Model saved: ./model_checkpoints/checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch))

# Else, loading the trained weights and performing generation only
elif WEIGHTS == '':
  # Loading the trained weights
  model.load_weights(WEIGHTS)
  generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
  print('\n\n')
else:
  print('\n\n[*] Nothing to do!')
