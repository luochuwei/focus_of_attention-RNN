#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 20/02/2016
#    Usage: utils
#
############################################
import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle

# set use gpu programatically
import theano.sandbox.cuda
def use_gpu(gpu_id):
    if gpu_id > -1:
        theano.sandbox.cuda.use("gpu" + str(gpu_id))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape, name):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1), name)

def init_gradws(shape, name):
    return theano.shared(floatX(np.zeros(shape)), name)

def init_bias(size, name):
    return theano.shared(floatX(np.zeros((size,))), name)

def save_model(f, model):
    ps = {}
    for p in model.params:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, "wb"))

def load_model(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params:
        p.set_value(ps[p.name])
    return model


def save_data_dic(i2w_path, w2i_path, i2w, w2i):
    pickle.dump(i2w, open(i2w_path, 'wb'))
    pickle.dump(w2i, open(w2i_path, 'wb'))

def load_data_dic(i2w_path, w2i_path):
    i2w = pickle.load(open(i2w_path, 'rb'))
    w2i = pickle.load(open(w2i_path, 'rb'))
    return i2w, w2i