import mph
mph.option('session', 'stand-alone')


from sko.GA import GA
from sko.tools import set_run_mode

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time

import os
os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)

import sys

k_matcustom0 = 96.2
k_matcustom1 = 0.3

def save_history(y,x):
    content = np.append(y, x)
    content = content.reshape(1, -1)
    if os.path.exists('Xy_history.csv'):
        with open('Xy_history.csv', 'a') as f:
            np.savetxt(f, content, delimiter=',')
    else:
        np.savetxt('Xy_history.csv', content, delimiter=',')

def opt(model, matrix_mat):
    model = model.java
    num = 0;
    Sel0 = [];
    Sel1 = [];
    for i in range(0, 20):
        for j in range(0, 20):
            num += 1
            if (matrix_mat[i, j].item() == 0):
                Sel0.extend(model.component("comp1").selection("box" + str(num)).entities())
            else:
               Sel1.extend(model.component("comp1").selection("box" + str(num)).entities())


    model.component("comp1").physics("ht2").feature("solidcustom1").selection().set(Sel0);
    model.component("comp1").physics("ht2").feature("solidcustom2").selection().set(Sel1);
    model.sol("sol1").runAll();

    probe_t = model.result().table("tbl1").getRealRow(0)[0] * 100
    # print("!!!!")
    return probe_t

# def client_create():


def worker(matrix_mat):

    # if not 'client' in dir():
    #     print("Clienting...")
    #     client = mph.start(cores=1)
    client = mph.start(cores=1)
    # if not 'model' in dir():
    #     print("Modeling...")
    #     model = client.load('start_star2.mph')
    model = client.load('start_star2.mph')
    temp = matrix_mat.reshape(20, 20)
    # print("Opting...")
    probe_t = opt(model, temp)
    print("probe_t:", probe_t)
    sys.stdout.flush()
    save_history(probe_t,matrix_mat)
    return probe_t

if __name__ == '__main__':
    iter = 0

    set_run_mode(worker, 'multiprocessing')
    # set_run_mode(worker, 'multithreading')
    ga = GA(func=worker, n_dim=400, size_pop=40, max_iter=1, prob_mut=0.001, lb=0, ub=1, precision=1)

    for i in range(200):
        start_time = time.time()
        print("Iter {} begins...".format(i+1))
        sys.stdout.flush()
        best_x, best_y = ga.run(1)
        print("probe_t:", best_y)
        print("Iter {} end used {:2.2f} sec(s), saving...".format(i+1, time.time()-start_time))
        sys.stdout.flush()

        Y_history = pd.DataFrame(ga.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')

        Y_history_gen = np.array(ga.generation_best_Y)
        X_history_gen = np.array(ga.generation_best_X)
        Y_history = np.array(ga.all_history_Y)

        np.savetxt('Y_history_gen.csv', Y_history_gen, delimiter=',')
        np.savetxt('X_history_gen.csv', X_history_gen, delimiter=',')
        np.savetxt('Y_history.csv', Y_history, delimiter=',')

        plt.savefig('./images/history.png', format='png')

    print('best_x:', best_x, '\n', 'best_y:', best_y)







