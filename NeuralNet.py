#!/usr/bin/env python
# coding: utf-8

# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
from time import time
from numpy import random


# https://adventuresinmachinelearning.com/stochastic-gradient-descent/


def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect


def f(x):
    return 1 / (1 + np.exp(-x))


def f_deriv(x):
    return f(x) * (1 - f(x))


def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b


def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

# Batch
def feed_forward_back_propagation(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise,
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l))
    return h, z


def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)


def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)


def train_nn_batch(nn_structure, X, y, iter_num=3000, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}

            h, z = feed_forward_back_propagation(X[i, :], W, b)
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                        
                        tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
                    
                        tri_b[l] += delta[l+1]
                    
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
            
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward_back_propagation(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y

# stochastic gradient descent. 

def train_nn_SGD(nn_structure, X, y, iter_num=3000, alpha=0.25, lamb=0.000):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%50 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, 
            # to be used in the gradient descent step
            h, z = feed_forward_back_propagation(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] = np.dot(delta[l+1][:,np.newaxis],
                                       np.transpose(h[l][:,np.newaxis])) 
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] = delta[l+1]
            # perform the gradient descent step for the weights in each layer
            for l in range(len(nn_structure) - 1, 0, -1):
                W[l] += -alpha * (tri_W[l] + lamb * W[l])
                b[l] += -alpha * (tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


def get_mini_batches(X, y, batch_size):
    random_idxs = random.choice(len(y), len(y), replace=False)
    X_shuffled = X[random_idxs,:]
    y_shuffled = y[random_idxs]
    mini_batches = [(X_shuffled[i:i+batch_size,:], y_shuffled[i:i+batch_size]) for
                   i in range(0, len(y), batch_size)]
    return mini_batches

# Mini batch gradient descent

def train_nn_MBGD(nn_structure, X, y, bs=100, iter_num=3000, alpha=0.25, lamb=0.000):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        mini_batches = get_mini_batches(X, y, bs)
        for mb in mini_batches:
            X_mb = mb[0]
            y_mb = mb[1]
            for i in range(len(y_mb)):
                delta = {}

                h, z = feed_forward_back_propagation(X_mb[i, :], W, b)
                # loop from nl-1 to 1 backpropagating the errors
                
                for l in range(len(nn_structure), 0, -1):
                    if l == len(nn_structure):
                        delta[l] = calculate_out_layer_delta(y_mb[i,:], h[l], z[l])
                        avg_cost += np.linalg.norm((y_mb[i,:]-h[l]))
                    else:
                        if l > 1:
                            delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                        # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                        tri_W[l] += np.dot(delta[l+1][:,np.newaxis], 
                                          np.transpose(h[l][:,np.newaxis])) 
                        # trib^(l) = trib^(l) + delta^(l+1)
                        tri_b[l] += delta[l+1]
            # perform the gradient descent step for the weights in each layer
            for l in range(len(nn_structure) - 1, 0, -1):
                W[l] += -alpha * (1.0/bs * tri_W[l] + lamb * W[l])
                b[l] += -alpha * (1.0/bs * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func



def main():
    iris = load_iris()
    X_scale = StandardScaler()
    X = X_scale.fit_transform(iris.data)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    y_v_train = convert_y_to_vect(y_train)
    y_v_test = convert_y_to_vect(y_test)
    
    nn_structure = [4, 15, 10]

    start = time()
    W, b, avg_cost_func = train_nn_SGD(nn_structure, X_train, y_v_train)
    end = time()
    
    plt.plot(avg_cost_func)
    plt.ylabel('Average J')
    plt.xlabel('Iteration number')
    plt.show()

    y_pred = predict_y(W, b, X_test, 3)
    print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
    print("Run time: " + str(end - start) + "s")

    
main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




