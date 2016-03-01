#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 11/02/2016
#    Usage: data processing
#
############################################


import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))



def print_sentence(sents, dim_y, i2w):
    for s in xrange(int(sents.shape[1] / dim_y)):
        xs = sents[:, s * dim_y : (s + 1) * dim_y]
        for w_i in xrange(xs.shape[0]):
            w = i2w[np.argmax(xs[w_i, :])]
            if w == "<eoss>":
                break
            elif w_i != 0 and w == "<soss>":
                break
            print w.decode('utf-8')," ",
        print "\n"





def processing(x_path, y_left_path, y_right_path, threshold, get_num_start, get_num_end, batch_size = 1):
    X_seqs = []
    yl_seqs = []
    yr_seqs = []
    i2w = {}
    w2i = {}
    lines_x = []
    lines_yl = []
    lines_yr = []
    tf = {}


    f_x = open(curr_path + "/" + x_path, "r")
    f_yl = open(curr_path + "/" + y_left_path, "r")
    f_yr = open(curr_path + "/" + y_right_path, "r")

    for line_x, line_yl, line_yr in zip(f_x, f_yl, f_yr):
        # print 'x  -->',line_x
        # print 'yl -->',line_yl
        # print 'yr -->',line_yr
        line_x = line_x.strip('\n')
        line_yl = line_yl.strip('\n')
        line_yr = line_yr.strip('\n')
        words_x = ['<soss>']+line_x.split()+["<eoss>"]
        # words_yl =  line_yl.split().reverse() + ['<soss>']
        words_yl = line_yl.split()
        if len(words_yl) > 0:
            words_yl.reverse()
        words_yl.append('<soss>')
        words_yr = line_yr.split() + ["<eoss>"]

        lines_x.append(words_x)
        lines_yl.append(words_yl)
        lines_yr.append(words_yr)

        for w in (words_x + words_yl + words_yr):
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
                tf[w] = 1
            else:
                tf[w] += 1
    f_x.flush()
    f_yl.flush()
    f_yr.flush()
    f_x.close()
    f_yl.close()
    f_yr.close()
    del f_x
    del f_yl
    del f_yr

    final_i2w = {}
    final_w2i = {}

    for word, num in tf.iteritems():
        if num > threshold:
            final_i2w[len(final_w2i)] = word
            final_w2i[word] = len(final_w2i)


    final_i2w[len(final_w2i)] = "<UNknown>"
    final_w2i["<UNknown>"] = len(final_w2i)

    i2w = final_i2w
    w2i = final_w2i
    # print len(i2w)
    del final_i2w
    del final_w2i

    assert len(lines_x) == len(lines_yl)
    assert len(lines_yl) == len(lines_yr)

    # for i in xrange(0, len(lines_x)):
    for i in xrange(get_num_start, get_num_end):
        # print i
        x_words = lines_x[i]
        yl_words = lines_yl[i]
        yr_words = lines_yr[i]

        x = np.zeros((len(x_words), len(w2i)), dtype = theano.config.floatX)
        yl = np.zeros((len(yl_words), len(w2i)), dtype = theano.config.floatX)
        yr = np.zeros((len(yr_words), len(w2i)), dtype = theano.config.floatX)

        

        for j in range(0, len(x_words)):
            if x_words[j] in w2i:
                x[j, w2i[x_words[j]]] = 1
            else:
                # x_n += 1
                x[j, w2i["<UNknown>"]] = 1
        for k in range(0, len(yl_words)):
            if yl_words[k] in w2i:
                yl[k, w2i[yl_words[k]]] = 1
            else:
                # yl_n += 1
                yl[k, w2i["<UNknown>"]] = 1
        for s in range(0, len(yr_words)):
            if yr_words[s] in w2i:
                yr[s, w2i[yr_words[s]]] = 1
            else:
                # yr_n += 1
                yr[s, w2i["<UNknown>"]] = 1
        
        X_seqs.append(x)
        yl_seqs.append(yl)
        yr_seqs.append(yr)
        del x
        del yl
        del yr

    data_x_yl_yr = batch_sequences(X_seqs, yl_seqs, yr_seqs, i2w, w2i, batch_size)

    return X_seqs, yl_seqs, yr_seqs, i2w, w2i, tf, data_x_yl_yr




def batch_sequences(x_seqs, yl_seqs, yr_seqs, i2w, w2i, batch_size):
    assert len(x_seqs) == len(yl_seqs)
    assert len(x_seqs) == len(yr_seqs)
    assert len(i2w) == len(w2i)

    data_x_yl_yr = {}
    batch_x = []
    batch_yl = []
    batch_yr = []
    x_seqs_len = []
    yl_seqs_len = []
    yr_seqs_len = []
    batch_id = 0
    dim = len(w2i)
    zeros_m = np.zeros((1, dim), dtype = theano.config.floatX)

    for i in xrange(len(x_seqs)):
        xs = x_seqs[i]
        yls = yl_seqs[i]
        yrs = yr_seqs[i]

        X = xs[0 : len(xs), ]
        YL = yls[0 : len(yls), ]
        YR = yrs[0 : len(yrs), ]

        batch_x.append(X)
        batch_yl.append(YL)
        batch_yr.append(YR)

        x_seqs_len.append(X.shape[0])
        yl_seqs_len.append(YL.shape[0])
        yr_seqs_len.append(YR.shape[0])

        if len(batch_x) == batch_size or (i == len(x_seqs) - 1):
            x_max_len = np.max(x_seqs_len)
            yl_max_len = np.max(yl_seqs_len)
            yr_max_len = np.max(yr_seqs_len)

            mask_x = np.zeros((x_max_len, len(batch_x)), dtype = theano.config.floatX)
            mask_yl = np.zeros((yl_max_len, len(batch_yl)), dtype = theano.config.floatX)
            mask_yr = np.zeros((yr_max_len, len(batch_yr)), dtype = theano.config.floatX)

            concat_X = np.zeros((x_max_len, len(batch_x) * dim), dtype = theano.config.floatX)
            concat_YL = np.zeros((yl_max_len, len(batch_yl) * dim), dtype = theano.config.floatX)
            concat_YR = np.zeros((yr_max_len, len(batch_yr) * dim), dtype = theano.config.floatX)

            assert len(batch_x) == len(batch_yl)
            assert len(batch_x) == len(batch_yr)
            
            for b_i in xrange(len(batch_x)):
                X = batch_x[b_i]
                YL = batch_yl[b_i]
                YR = batch_yr[b_i]

                mask_x[0 : X.shape[0], b_i] = 1
                mask_yl[0 : YL.shape[0], b_i] = 1
                mask_yr[0 : YR.shape[0], b_i] = 1

                for r in xrange(x_max_len - X.shape[0]):
                    X = np.concatenate((X, zeros_m), axis=0)
                for rr in xrange(yl_max_len - YL.shape[0]):
                    YL = np.concatenate((YL, zeros_m), axis=0)
                for rrr in xrange(yr_max_len - YR.shape[0]):
                    YR = np.concatenate((YR, zeros_m), axis=0)
                concat_X[:, b_i * dim : (b_i + 1) * dim] = X
                concat_YL[:, b_i * dim : (b_i + 1) * dim] = YL
                concat_YR[:, b_i * dim : (b_i + 1) * dim] = YR

            data_x_yl_yr[batch_id] = [concat_X, mask_x, concat_YL, mask_yl, concat_YR, mask_yr, len(batch_x)]
            batch_x = []
            batch_yl = []
            batch_yr = []
            x_seqs_len = []
            yl_seqs_len = []
            yr_seqs_len = []
            batch_id += 1

    return data_x_yl_yr




# x_path = "data/X.txt"
# y_left_path = "data/y_left.txt"
# y_right_path = "data/y_right.txt"
# batch_size = 2


# X_seqs, yl_seqs, yr_seqs, i2w, w2i, data_x_yl_yr = processing(x_path, y_left_path, y_right_path, batch_size)

# # l,r =processing(x_path, y_left_path, y_right_path, batch_size)
