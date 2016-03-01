#-*- coding:utf-8 -*-
#######################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 11/02/2016
#    Usage: Focus of attention encoder decoder network
#
#######################################################
import numpy as np
import theano
import theano.tensor as T

from gru import *
from lstm import *
from w_encoder import *
from s_encoder import *
from s_decoder import *
from attention import *
from w_decoder import *
from updates import *

class FANN(object):
    def __init__(self, in_size, out_size, hidden_size,
                 cell = "gru", optimizer = "rmsprop", p = 0.5, num_sents = 1):

        self.X = T.matrix("X")
        self.Y_left = T.matrix("Y_left")
        self.Y_right = T.matrix("Y_right")
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.drop_rate = p
        self.num_sents = num_sents
        self.is_train = T.iscalar('is_train') # for dropout
        self.batch_size = T.iscalar('batch_size') # for mini-batch training
        self.mask = T.matrix("mask")
        self.mask_y_left = T.matrix("mask_y_left")
        self.mask_y_right = T.matrix("mask_y_right")
        self.optimizer = optimizer
        self.define_layers()
        self.define_train_test_funcs()
                
    def define_layers(self):
        self.layers = []
        self.params = []
        rng = np.random.RandomState(1234)
        # LM layers
        word_encoder_layer = WordEncoderLayer(rng, self.X, self.in_size, self.out_size, self.hidden_size,
                         self.cell, self.optimizer, self.drop_rate,
                         self.is_train, self.batch_size, self.mask)
        self.layers += word_encoder_layer.layers
        self.params += word_encoder_layer.params

        i = len(self.layers) - 1

        # encoder layer
        layer_input = word_encoder_layer.activation
        self.test = word_encoder_layer.activation
        encoder_layer = SentEncoderLayer(self.cell, rng, str(i + 1), (word_encoder_layer.hidden_size, word_encoder_layer.hidden_size),
                                         layer_input, self.mask, self.is_train, self.batch_size, self.drop_rate)
        self.layers.append(encoder_layer)
        self.params += encoder_layer.params
        
        # codes is a vector
        codes = encoder_layer.activation
        self.test2 = encoder_layer.activation
        codes = T.reshape(codes, (1, encoder_layer.out_size))
        self.test3 = codes

        # sentence decoder
        # sent_decoder_layer = SentDecoderLayer(self.cell, rng, str(i + 2), (encoder_layer.out_size, encoder_layer.in_size),
        #                                  codes, self.mask, self.is_train, self.batch_size, self.drop_rate)
        sent_decoder_layer = SentDecoderLayer(self.cell, rng, str(i + 2), (encoder_layer.out_size, encoder_layer.in_size),
                                         codes, self.mask, self.is_train, self.batch_size, self.drop_rate)
        self.layers.append(sent_decoder_layer)
        self.params += sent_decoder_layer.params

        # attention layer (syncrhonous update)
        sent_encs = encoder_layer.sent_encs
        self.test4 = sent_encs
        sent_decs = sent_decoder_layer.activation
        self.test5 = sent_decs
        attention_layer = AttentionLayer(str(i + 3), (self.num_sents, sent_decoder_layer.out_size), sent_encs, sent_decs)
        
        # reshape to a row with num_sentences samples
        sents_codes = attention_layer.activation
        self.test6 = attention_layer.activation
        sents_codes = T.reshape(sents_codes, (1, self.batch_size * sent_decoder_layer.out_size))
        self.test7 = sents_codes

        # word decoder
        # word_decoder_layer = WordDecoderLayer(self.cell, rng, str(i + 4), (sent_decoder_layer.out_size, self.out_size),
        #                                  sents_codes, self.mask, self.is_train, self.batch_size, self.drop_rate)
        word_decoder_layer = WordDecoderLayer(self.cell, rng, str(i + 4), (sent_decoder_layer.out_size, self.out_size),
                                         sents_codes, self.mask_y_left, self.mask_y_right, self.is_train, self.batch_size, self.drop_rate)
        self.layers.append(word_decoder_layer)
        self.params += word_decoder_layer.params

        self.activation_left = word_decoder_layer.activation_left
        self.activation_right = word_decoder_layer.activation_right
        # self.test8 = word_decoder_layer.hhhh
        self.test8 = word_decoder_layer.hl
        self.test9 = word_decoder_layer.hr

        self.predict = theano.function(inputs = [self.X, self.mask, self.mask_y_left, self.mask_y_right, self.batch_size],
                                               givens = {self.is_train : np.cast['int32'](1)},
                                               outputs = [self.activation_left, self.activation_right],on_unused_input='ignore')


    # https://github.com/fchollet/keras/pull/9/files
        self.epsilon = 1.0e-15
    def categorical_crossentropy(self, y_pred, y_true, mask_y):
        y_pred = T.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        m = T.reshape(mask_y, (mask_y.shape[0] * self.batch_size, 1))
        ce = T.nnet.categorical_crossentropy(y_pred, y_true)
        ce = T.reshape(ce, (mask_y.shape[0] * self.batch_size, 1))
        return T.sum(ce * m) / T.sum(m)
        # return ce.mean()


    def define_train_test_funcs(self):
        pYs_left = T.reshape(self.activation_left, (self.mask_y_left.shape[0] * self.batch_size, self.out_size))
        pYs_right = T.reshape(self.activation_right, (self.mask_y_right.shape[0] * self.batch_size, self.out_size))
        # tYs =  T.reshape(self.X, (self.mask.shape[0] * self.batch_size, self.out_size))
        tYs_left =  T.reshape(self.Y_left, (self.mask_y_left.shape[0] * self.batch_size, self.out_size))
        tYs_right =  T.reshape(self.Y_right, (self.mask_y_right.shape[0] * self.batch_size, self.out_size))
        cost_left = self.categorical_crossentropy(pYs_left, tYs_left, self.mask_y_left)
        cost_right = self.categorical_crossentropy(pYs_right, tYs_right, self.mask_y_right)

        cost = cost_left + cost_right

        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        lr = T.scalar("lr")
        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)

        #updates = sgd(self.params, gparams, lr)
        #updates = momentum(self.params, gparams, lr)
        #updates = rmsprop(self.params, gparams, lr)
        #updates = adagrad(self.params, gparams, lr)
        #updates = adadelta(self.params, gparams, lr)
        #updates = adam(self.params, gparams, lr)
        
        self.train = theano.function(inputs = [self.X, self.Y_left, self.Y_right, self.mask, self.mask_y_left, self.mask_y_right, lr, self.batch_size],
                                               givens = {self.is_train : np.cast['int32'](1)},
                                               outputs = [cost, self.activation_left, self.activation_right, self.test, self.test2, self.test3, self.test4, self.test5, self.test6, self.test7, self.test8, self.test9],
                                               updates = updates)
