import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GrdDscnt import grdescentnormal
from GrdDscntQuant import grdescentquant
from NormalLog import normallogistic
from QuantLog import quantlogistic
from sklearn.model_selection import train_test_split



def test_loss(w,X,y):
    #calculates test loss
    log_odds = np.dot(w.T, X)
    #change place
    exp_values = np.exp(-np.clip(log_odds, -500, 500))
    
    probs = 1 / (1 + exp_values)
    
    preds = (probs > 0.5).astype(int)
    preds = np.where(preds == 0, -1, preds)
    test_loss = np.sum(preds != y) / len(y)

    return test_loss

def experiment(X,y):
    # remember to transpose data to have shape dxn
    quantizers = ['unif', 'gauss']
    scales = ['squa', 1/2,2,2**0.5]
    levels = [1,2,3,4]
    func = quantlogistic
    w0 = np.random.uniform(-1, 1, (X.shape[0], 1))
    for scale in scales:

        for type_w in quantizers:
            if type_w=='unif':
                w_level = [1,2,3,4,5,6,7,8]
            elif type_w == 'gauss':
                w_level = [1,2,3,4]
            for type_q in quantizers:
            
                if type_q == 'unif':
                    q_level = [1,2,3,4,5,6,7,8]
                elif type_q == 'gauss':
                    q_level = [1,2,3,4]
                loss_grid = np.zeros((len(w_level),len(q_level)))
                for i, level_w in enumerate(w_level):
                    for j, level_q in enumerate(q_level):
                        w, iters = grdescentquant(func, w0, 0.1, 10, X, y, level_w, level_q, type_w, type_q, scale,tolerance=1e-02)
                        loss = test_loss(w,X,y)
                        loss_grid[i,j] = loss
                xlabel = "gradient lvl"
                ylabel = "w lvl"
                plt.pcolormesh(loss_grid)
                plt.colorbar()
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(f"w quantizer: {type_w}, gradient quantizer: {type_q},scale: {scale}")
                plt.show()
                print(loss_grid)






"""def experiment(X,y, bins: list):


    generates plots to compare performance of quantized gradient with normal gradient accross different number of bins
    :param X: features
    :param y: labels
    :param num_trials:
    :param bins: array of number of bins to try
    :return: normal_iters, quant_iters w_quant, w



    normal_iters = []
    quant_iters = []
    normal_loss = []
    quant_loss = []
    w_quants = []

    # split randomly into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # have to transpose data to get it to work with the function implementation where features are along the rows
    X_train = X_train.T
    X_test = X_test.T
    y_test = y_test.T
    y_train = y_train.T
    w0 = np.random.uniform(-1, 1, (X_train.shape[0], 1))
    w,iters = grdescentnormal(normallogistic, w0, 0.1, 50000, X_train, y_train)
    loss = test_loss(w, X_test, y_test)
    print(loss)

    # store the results
    normal_iters.append(iters)
    normal_loss.append(loss)

    #do the same for quantized version
    for num_bin in bins:
        w_quant, iters = grdescentquant(quantlogistic, w0, 0.1, 50000, X_train, y_train, num_bin)
        loss = test_loss(w_quant,X_test,y_test)

        quant_iters.append((num_bin, iters))
        quant_loss.append((num_bin, loss))
    iters_dict = dict(quant_iters)
    loss_dict = dict(quant_loss)
    iters_dict = {key: np.mean(values) for key, values in iters_dict.items()}
    loss_dict = {key: np.mean(values) for key, values in loss_dict.items()}
    loss_dict[0] = np.mean(normal_loss)

    plt.bar(loss_dict.keys(), loss_dict.values())
    plt.xlabel("log bins")
    plt.ylabel("mean test loss")
    plt.show()

    plt.bar(iters_dict.keys(), iters_dict.values())
    plt.xlabel("log bins")
    plt.ylabel("mean iterations until convergence")
    plt.show()


    return normal_iters, quant_iters,  w_quant, w"""