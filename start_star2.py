import mph

import model_opt as models

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



# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

k_matcustom0 = 96.2
k_matcustom1 = 0.3
matrix_mat = np.random.rand(10,10)


# def setup_module(k_matcustom0, k_matcustom1,matrix_mat):
#     # global client
#     # client.java.initStandalone(JBoolean(False))
#
#     model, probe_t = models.generate(client, k_matcustom0, k_matcustom1,matrix_mat)
#     return model, probe_t
#

def save_history(y,x):
    global Xy_history
    if (Xy_history.size):
        Xy_history = np.vstack((Xy_history, np.append(y, x)))
    else:
        Xy_history = np.append(y, x)


def train_module(model, matrix_mat):

    probe_t = models.opt(model, matrix_mat)
    return probe_t


def worker(matrix_mat):


    temp = matrix_mat.reshape(20, 20)
    probe_t = train_module(model, temp)
    print("probe_t:", probe_t)
    save_history(probe_t, matrix_mat)


    return probe_t

def save_model(name):

    model.save("./models/"+name)

# if __name__ == '__main__':
#     iter = 0
#     Xy_history = np.array([])
#     # set_run_mode(worker, 'multiprocessing')
#     # set_run_mode(worker, 'multithreading')
#     # set_run_mode(func, 'cached')
#     # pso = PSO(func=worker, n_dim=400,pop=5, max_iter=1, lb=0, ub=1)
#
#     mph.option('session', 'stand-alone')
#     client = mph.start()
#
#     print("Loading...")
#     model = client.load('start_star2.mph')
#     print("Modeling...")
#
#     ga = GA(func=worker, n_dim=400, size_pop=500, max_iter=1, prob_mut=0.001, lb=0, ub=1, precision=1)
#
#     for i in range(100):
#         start_time = time.time(1)
#         print("Iter {} begins...".format(i+1))
#         best_x, best_y = ga.run(1)
#         # best_x, best_y = pso.run(1)
#         print("Iter {} end used {:2.2f} sec(s), saving...".format(i+1, time.time()-start_time))
#         # save_model("iter_"+str(i+1))
#
#         Y_history = pd.DataFrame(ga.all_history_Y)
#         fig, ax = plt.subplots(2, 1)
#         ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
#         Y_history.min(axis=1).cummin().plot(kind='line')
#
#         # # plt.show()
#         Y_history_gen = np.array(ga.generation_best_Y)
#         X_history_gen = np.array(ga.generation_best_X)
#         Y_history = np.array(ga.all_history_Y)
#         # Xy_history_np = np.array(Xy_history)
#
#         np.save('Y_history_gen.npy', Y_history_gen)
#         np.save('X_history_gen.npy', X_history_gen)
#         np.save('Y_history.npy', Y_history)
#         np.save('Xy_history_np.npy', Xy_history)
#
#         plt.savefig('./images/history.png', format='png')
#
#     # print('best_x is ', pso.gbest_x)
#     # print('best_y is', pso.gbest_y)
#     # plt.plot(pso.gbest_y_hist)
#
#     print('best_x:', best_x, '\n', 'best_y:', best_y)
#     # plt.plot(pso.gbest_y_hist)





    # for i in range(1,6):
    #     # model.result().export().create("t"+str(i), "pg"+str(i), "Image2D")
    #     # filename = "./t"+str(i-2)+".png"
    #     # print(filename)
    #     # model.result().export("t"+str(i)).set("filename", filename)
    #     # model.result().export("t"+str(i)).run()
    #
        # model.result().export().create("t"+str(i), "pg"+str(i), "Image2D")
        # model.result().export("t"+str(i)).set("filename", "./t"+str(i)+".png")
        # model.result().export("t"+str(i)).run()






