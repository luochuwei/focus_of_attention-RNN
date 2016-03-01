#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 19/02/2016
#    Usage: prediction
#
############################################

import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from focus_of_attention_NN import *
import data
import test_data




def predict():
	pass



e = 0.01   #error
lr = 0.5
drop_rate = 0.
batch_size = 1   #must be same as the loading model's batch size
hidden_size = [500]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adadelta" 


test_path = "data/test.txt"
test_data = data.word_sequence(test_path, batch_size, left)

i2w, w2i = load_data_dic("data/i2w.pkl", "data/w2i.pkl")

dim_x = len(w2i)
dim_y = len(w2i)

num_sents = batch_size