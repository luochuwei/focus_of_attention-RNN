#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 11/02/2016
#    Usage: decoder
#
############################################

import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from lstm import *
from gru import *

class WordDecoderLayer(object):
    def __init__(self, cell, rng, layer_id, shape, X, mask_left, mask_right, is_train = 1, batch_size = 1, p = 0.5):
        prefix = "WordDecoderLayer_"
        layer_id = "_" + layer_id
        self.out_size, self.in_size = shape
        self.mask_left = mask_left
        self.mask_right = mask_right
        self.X = X  # sentence code
        self.words_left = mask_left.shape[0]
        self.words_right = mask_right.shape[0]

        
        self.W_hy = init_weights((self.out_size, self.in_size), prefix + "W_hy" + layer_id)
        self.b_y = init_bias(self.in_size, prefix + "b_y" + layer_id)
        self.W_hy_right = init_weights((self.out_size, self.in_size), prefix + "W_hy_right" + layer_id)
        self.b_y_right = init_bias(self.in_size, prefix + "b_y_right" + layer_id)
        
        self.W_left_to_right = init_weights((self.out_size, self.out_size), prefix + "W_left_to_right" + layer_id)
        if cell == "gru":
            #left decoder
            self.decoder = GRULayer(rng, prefix + layer_id, (self.in_size, self.out_size), self.X, mask_left, is_train, batch_size, p)
            def _active(m, pre_h, x):
                x = T.reshape(x, (batch_size, self.in_size))
                pre_h = T.reshape(pre_h, (batch_size, self.out_size))

                h = self.decoder._active(x, pre_h)
                y = T.nnet.softmax(T.dot(h, self.W_hy) + self.b_y)
                y = y * m[:, None]

                h = T.reshape(h, (1, batch_size * self.out_size))
                y = T.reshape(y, (1, batch_size * self.in_size))
                return h, y
            [h_left, y_left], updates_left = theano.scan(_active, #n_steps = self.words,
                                      sequences = [self.mask_left],
                                      outputs_info = [{'initial':self.X, 'taps':[-1]},
                                      T.alloc(floatX(0.), 1, batch_size * self.in_size)])

            #right decoder
            h_left_last = T.reshape(h_left[h_left.shape[0]-1, :], (batch_size, self.out_size))
            right_input_r = T.dot(h_left_last, self.W_left_to_right)
            right_input_r = T.reshape(right_input_r, (1, batch_size*self.out_size))
            right_input = h_left[0,:] + right_input_r

            self.decoder_right = GRULayer(rng, prefix + "_right_" + layer_id, (self.in_size, self.out_size), right_input, mask_right, is_train, batch_size, p)

            def _active_right(m, pre_h, x):
                x = T.reshape(x, (batch_size, self.in_size))
                pre_h = T.reshape(pre_h, (batch_size, self.out_size))

                h = self.decoder_right._active(x, pre_h)
                y = T.nnet.softmax(T.dot(h, self.W_hy_right) + self.b_y_right)
                y = y * m[:, None]

                h = T.reshape(h, (1, batch_size * self.out_size))
                y = T.reshape(y, (1, batch_size * self.in_size))
                return h, y
            
            [h_right, y_right], updates_right = theano.scan(_active_right, #n_steps = self.words,
                                      sequences = [self.mask_right],
                                      outputs_info = [{'initial':right_input, 'taps':[-1]},
                                      T.alloc(floatX(0.), 1, batch_size * self.in_size)])

        elif cell == "lstm":
            # self.decoder = LSTMLayer(rng, prefix + layer_id, (self.in_size, self.out_size), self.X, mask, is_train, batch_size, p)
            # def _active(m, pre_h, pre_c, x):
            #     x = T.reshape(x, (batch_size, self.in_size))
            #     pre_h = T.reshape(pre_h, (batch_size, self.out_size))
            #     pre_c = T.reshape(pre_c, (batch_size, self.out_size))

            #     h, c = self.decoder._active(x, pre_h, pre_c)
            
            #     y = T.nnet.softmax(T.dot(h, self.W_hy) + self.b_y)
            #     y = y * m[:, None]

            #     h = T.reshape(h, (1, batch_size * self.out_size))
            #     c = T.reshape(c, (1, batch_size * self.out_size))
            #     y = T.reshape(y, (1, batch_size * self.in_size))
            #     return h, c, y
            # [h, c, y], updates = theano.scan(_active, #n_steps = self.words,
            #                                  sequences = [self.mask],
            #                                  outputs_info = [{'initial':self.X, 'taps':[-1]},
            #                                                  {'initial':self.X, 'taps':[-1]},
            #                                                  T.alloc(floatX(0.), 1, batch_size * self.in_size)])
            print "lstm error"
        
        y_left = T.reshape(y_left, (self.words_left, batch_size * self.in_size))
        y_right = T.reshape(y_right, (self.words_right, batch_size * self.in_size))
        self.activation_left = y_left
        self.activation_right = y_right
        self.params = self.decoder.params + self.decoder_right.params + [self.W_hy, self.b_y, self.W_hy_right, self.b_y_right, self.W_left_to_right]
        self.hl = h_left
        self.hr = h_right