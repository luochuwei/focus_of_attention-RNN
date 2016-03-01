#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 11/02/2016
#    Usage: test data processing
#
############################################
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip


curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


def word_sequence(f_path, batch_size = 1, i2w, w2i):
    test_seqs = []
    lines = []
    tf = {}
    f = open(curr_path + "/" + f_path, "r")
    for line in f:
        line = line.strip('\n').lower()
        words = ['<soss>']+line_x.split()+["<eoss>"]
        lines.append(words)
    f.close()

    for i in range(0, len(lines)):
        words = lines[i]
        x = np.zeros((len(words), len(w2i)), dtype = theano.config.floatX)
        for j in range(0, len(words)):
            if words[j] in w2i:
                x[j, w2i[words[j]]] = 1
        test_seqs.append(np.asmatrix(x))

    test_data_x = batch_sequences(test_seqs, i2w, w2i, batch_size)

    return test_seqs, i2w, w2i, test_data_x

def batch_sequences(seqs, i2w, w2i, batch_size):
    test_data_x = {}
    batch_x = []
    # batch_y = []
    seqs_len = []
    batch_id = 0
    dim = len(w2i)
    zeros_m = np.zeros((1, dim), dtype = theano.config.floatX)
    for i in xrange(len(seqs)):
        seq = seqs[i];
        X = seq[0 : len(seq), ]
        batch_x.append(X)
        seqs_len.append(X.shape[0])


        if len(batch_x) == batch_size or (i == len(seqs) - 1):
            max_len = np.max(seqs_len);
            mask = np.zeros((max_len, len(batch_x)), dtype = theano.config.floatX)
            
            concat_X = np.zeros((max_len, len(batch_x) * dim), dtype = theano.config.floatX)

            for b_i in xrange(len(batch_x)):
                X = batch_x[b_i]
                mask[0 : X.shape[0], b_i] = 1
                for r in xrange(max_len - X.shape[0]):
                    X = np.concatenate((X, zeros_m), axis=0)

                concat_X[:, b_i * dim : (b_i + 1) * dim] = X 

            test_data_x[batch_id] = [concat_X, mask, len(batch_x)]
            batch_x = []

            seqs_len = []
            batch_id += 1
    return test_data_x

